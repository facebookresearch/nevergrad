# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import math
import logging
import itertools
from collections import deque
import warnings
import cma
import numpy as np
from bayes_opt import UtilityFunction
from bayes_opt import BayesianOptimization
import nevergrad.common.typing as tp
from nevergrad.common import errors
from nevergrad.parametrization import parameter as p
from nevergrad.parametrization import transforms
from nevergrad.parametrization import discretization
from nevergrad.parametrization import _layering
from nevergrad.parametrization import _datalayers
from . import oneshot
from . import base
from . import mutations
from .base import registry as registry
from .base import addCompare  # pylint: disable=unused-import
from .base import IntOrParameter


# families of optimizers
# pylint: disable=unused-wildcard-import,wildcard-import,too-many-lines,too-many-arguments
from .differentialevolution import *  # type: ignore  # noqa: F403
from .es import *  # type: ignore  # noqa: F403
from .oneshot import *  # noqa: F403
from .recastlib import *  # noqa: F403

try:
    from .externalbo import HyperOpt  # pylint: disable=unused-import
except ModuleNotFoundError:
    pass

# run with LOGLEVEL=DEBUG for more debug information
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

# # # # # optimizers # # # # #


class _OnePlusOne(base.Optimizer):
    """Simple but sometimes powerful optimization algorithm.

    We use the one-fifth adaptation rule, going back to Schumer and Steiglitz (1968).
    It was independently rediscovered by Devroye (1972) and Rechenberg (1973).
    We use asynchronous updates, so that the 1+1 can actually be parallel and even
    performs quite well in such a context - this is naturally close to 1+lambda.

    Posssible mutations include gaussian and cauchy for the continuous case, and in the discrete case:
    discrete, fastga, doublefastga, adaptive, portfolio, discreteBSO, doerr.
    - discrete is the most classical discrete mutation operator,
    - doubleFastGA is an adaptation of FastGA to arity > 2, Portfolio corresponds to random mutation rates,
    - discreteBSO corresponds to a decreasing schedule of mutation rate.
    - adaptive and doerr correspond to various self-adaptive mutation rates.
    - coordinatewise_adaptive is the anisotropic counterpart of the adaptive version.
    """

    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        *,
        noise_handling: tp.Optional[tp.Union[str, tp.Tuple[str, float]]] = None,
        mutation: str = "gaussian",
        crossover: bool = False,
        use_pareto: bool = False,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._sigma: float = 1
        self._previous_best_loss = float("inf")
        self.use_pareto = use_pareto
        all_params = p.helpers.flatten(self.parametrization)
        arity = max(
            len(param.choices) if isinstance(param, p.TransitionChoice) else 500 for _, param in all_params
        )
        self.arity_for_discrete_mutation = arity
        # configuration
        if noise_handling is not None:
            if isinstance(noise_handling, str):
                assert noise_handling in [
                    "random",
                    "optimistic",
                ], f"Unkwnown noise handling: '{noise_handling}'"
            else:
                assert isinstance(
                    noise_handling, tuple
                ), "noise_handling must be a string or  a tuple of type (strategy, factor)"
                assert noise_handling[1] > 0.0, "the factor must be a float greater than 0"
                assert noise_handling[0] in [
                    "random",
                    "optimistic",
                ], f"Unkwnown noise handling: '{noise_handling}'"
        assert mutation in [
            "gaussian",
            "cauchy",
            "discrete",
            "fastga",
            "doublefastga",
            "adaptive",
            "coordinatewise_adaptive",
            "portfolio",
            "discreteBSO",
            "lengler",
            "doerr",
        ], f"Unkwnown mutation: '{mutation}'"
        if mutation == "adaptive":
            self._adaptive_mr = 0.5
        if mutation == "coordinatewise_adaptive":
            self._velocity = self._rng.uniform(size=self.dimension) * arity / 4.0
            self._modified_variables = np.array([True] * self.dimension)
        self.noise_handling = noise_handling
        self.mutation = mutation
        self.crossover = crossover
        if mutation == "doerr":
            assert num_workers == 1, "Doerr mutation is implemented only in the sequential case."
            self._doerr_mutation_rates = [1, 2]
            self._doerr_mutation_rewards = [0.0, 0.0]
            self._doerr_counters = [0.0, 0.0]
            self._doerr_epsilon = 0.25  # self.dimension ** (-0.01)
            self._doerr_gamma = 1 - 2 / self.dimension
            self._doerr_current_best = float("inf")
            i = 3
            j = 2
            self._doerr_index: int = -1  # Nothing has been mutated for now.
            while i < self.dimension:
                self._doerr_mutation_rates += [i]
                self._doerr_mutation_rewards += [0.0]
                self._doerr_counters += [0.0]
                i += j
                j += 2

    def _internal_ask_candidate(self) -> p.Parameter:
        # pylint: disable=too-many-return-statements, too-many-branches
        noise_handling = self.noise_handling
        if not self._num_ask:
            out = self.parametrization.spawn_child()
            out._meta["sigma"] = self._sigma
            return out
        # for noisy version
        if noise_handling is not None:
            limit = (0.05 if isinstance(noise_handling, str) else noise_handling[1]) * len(self.archive) ** 3
            strategy = noise_handling if isinstance(noise_handling, str) else noise_handling[0]
            if self._num_ask <= limit:
                if strategy in ["cubic", "random"]:
                    idx = self._rng.choice(len(self.archive))
                    return list(self.archive.values())[idx].parameter.spawn_child()  # type: ignore
                elif strategy == "optimistic":
                    return self.current_bests["optimistic"].parameter.spawn_child()
        # crossover
        mutator = mutations.Mutator(self._rng)
        pessimistic = self.current_bests["pessimistic"].parameter.spawn_child()
        if self.num_objectives > 1 and self.use_pareto:  # multiobjective
            # revert to using a sample of the pareto front (not "pessimistic" though)
            pareto = (
                self.pareto_front()
            )  # we can't use choice directly, because numpy does not like iterables
            pessimistic = pareto[self._rng.choice(len(pareto))].spawn_child()
        ref = self.parametrization
        if self.crossover and self._num_ask % 2 == 1 and len(self.archive) > 2:
            data = mutator.crossover(
                pessimistic.get_standardized_data(reference=ref), mutator.get_roulette(self.archive, num=2)
            )
            return pessimistic.set_standardized_data(data, reference=ref)
        # mutating

        mutation = self.mutation
        if mutation in ("gaussian", "cauchy"):  # standard case
            step = (
                self._rng.normal(0, 1, self.dimension)
                if mutation == "gaussian"
                else self._rng.standard_cauchy(self.dimension)
            )
            out = pessimistic.set_standardized_data(self._sigma * step)
            out._meta["sigma"] = self._sigma
            return out
        else:
            pessimistic_data = pessimistic.get_standardized_data(reference=ref)
            if mutation == "crossover":
                if self._num_ask % 2 == 0 or len(self.archive) < 3:
                    data = mutator.portfolio_discrete_mutation(
                        pessimistic_data, arity=self.arity_for_discrete_mutation
                    )
                else:
                    data = mutator.crossover(pessimistic_data, mutator.get_roulette(self.archive, num=2))
            elif mutation == "adaptive":
                data = mutator.portfolio_discrete_mutation(
                    pessimistic_data,
                    intensity=max(1, int(self._adaptive_mr * self.dimension)),
                    arity=self.arity_for_discrete_mutation,
                )
            elif mutation == "discreteBSO":
                assert self.budget is not None, "DiscreteBSO needs a budget."
                intensity = int(self.dimension - self._num_ask * self.dimension / self.budget)
                if intensity < 1:
                    intensity = 1
                data = mutator.portfolio_discrete_mutation(
                    pessimistic_data, intensity=intensity, arity=self.arity_for_discrete_mutation
                )
            elif mutation == "coordinatewise_adaptive":
                self._modified_variables = np.array([True] * self.dimension)
                data = mutator.coordinatewise_mutation(
                    pessimistic_data,
                    self._velocity,
                    self._modified_variables,
                    arity=self.arity_for_discrete_mutation,
                )
            elif mutation == "lengler":
                alpha = 1.54468
                intensity = int(max(1, self.dimension * (alpha * np.log(self.num_ask) / self.num_ask)))
                data = mutator.portfolio_discrete_mutation(
                    pessimistic_data, intensity=intensity, arity=self.arity_for_discrete_mutation
                )
            elif mutation == "doerr":
                # Selection, either random, or greedy, or a mutation rate.
                assert self._doerr_index == -1, "We should have used this index in tell."
                if self._rng.uniform() < self._doerr_epsilon:
                    index = self._rng.choice(range(len(self._doerr_mutation_rates)))
                    self._doerr_index = index
                else:
                    index = self._doerr_mutation_rewards.index(max(self._doerr_mutation_rewards))
                    self._doerr_index = -1
                intensity = self._doerr_mutation_rates[index]
                data = mutator.portfolio_discrete_mutation(
                    pessimistic_data, intensity=intensity, arity=self.arity_for_discrete_mutation
                )
            else:
                func: tp.Any = {  # type: ignore
                    "discrete": mutator.discrete_mutation,
                    "fastga": mutator.doerr_discrete_mutation,
                    "doublefastga": mutator.doubledoerr_discrete_mutation,
                    "portfolio": mutator.portfolio_discrete_mutation,
                }[mutation]
                data = func(pessimistic_data, arity=self.arity_for_discrete_mutation)
            return pessimistic.set_standardized_data(data, reference=ref)

    def _internal_tell(self, x: tp.ArrayLike, loss: tp.FloatLoss) -> None:
        # only used for cauchy and gaussian
        if self._previous_best_loss != loss:
            self._sigma *= 2.0 if loss < self._previous_best_loss else 0.84
        if self.mutation == "doerr" and self._doerr_current_best < float("inf") and self._doerr_index >= 0:
            improvement = max(0.0, self._doerr_current_best - loss)
            # Decay.
            index = self._doerr_index
            counter = self._doerr_counters[index]
            self._doerr_mutation_rewards[index] = (
                self._doerr_gamma * counter * self._doerr_mutation_rewards[index] + improvement
            ) / (self._doerr_gamma * counter + 1)
            self._doerr_counters = [self._doerr_gamma * x for x in self._doerr_counters]
            self._doerr_counters[index] += 1
            self._doerr_index = -1
        if self.mutation == "doerr":
            self._doerr_current_best = min(self._doerr_current_best, loss)
        if self.mutation == "adaptive":
            factor = 1.2 if loss <= self._previous_best_loss else 0.731  # 0.731 = 1.2**(-np.exp(1)-1)
            self._adaptive_mr = min(1.0, factor * self._adaptive_mr)
        if self.mutation == "coordinatewise_adaptive":
            factor = 1.2 if loss < self._previous_best_loss else 0.731  # 0.731 = 1.2**(-np.exp(1)-1)
            inds = self._modified_variables
            self._velocity[inds] = np.clip(
                self._velocity[inds] * factor, 1.0, self.arity_for_discrete_mutation / 4.0
            )
        self._previous_best_loss = self.current_bests["pessimistic"].mean  # could be the current one


class ParametrizedOnePlusOne(base.ConfiguredOptimizer):
    """Simple but sometimes powerfull class of optimization algorithm.
    This use asynchronous updates, so that (1+1) can actually be parallel and even
    performs quite well in such a context - this is naturally close to (1+lambda).


    Parameters
    ----------
    noise_handling: str or Tuple[str, float]
        Method for handling the noise. The name can be:

        - `"random"`: a random point is reevaluated regularly, this uses the one-fifth adaptation rule,
          going back to Schumer and Steiglitz (1968). It was independently rediscovered by Devroye (1972) and Rechenberg (1973).
        - `"optimistic"`: the best optimistic point is reevaluated regularly, optimism in front of uncertainty
        - a coefficient can to tune the regularity of these reevaluations (default .05)
    mutation: str
        One of the available mutations from:

        - `"gaussian"`: standard mutation by adding a Gaussian random variable (with progressive
          widening) to the best pessimistic point
        - `"cauchy"`: same as Gaussian but with a Cauchy distribution.
        - `"discrete"`: when a variable is mutated (which happens with probability 1/d in dimension d), it's just
             randomly drawn. This means that on average, only one variable is mutated.
        - `"discreteBSO"`: as in brainstorm optimization, we slowly decrease the mutation rate from 1 to 1/d.
        - `"fastga"`: FastGA mutations from the current best
        - `"doublefastga"`: double-FastGA mutations from the current best (Doerr et al, Fast Genetic Algorithms, 2017)
        - `"portfolio"`: Random number of mutated bits (called niform mixing in
          Dang & Lehre "Self-adaptation of Mutation Rates in Non-elitist Population", 2016)
        - `"lengler"`: specific mutation rate chosen as a function of the dimension and iteration index.
    crossover: bool
        whether to add a genetic crossover step every other iteration.
    use_pareto: bool
        whether to restart from a random pareto element in multiobjective mode, instead of the last one added

    Notes
    -----
    After many papers advocared the mutation rate 1/d in the discrete (1+1) for the discrete case,
    `it was proposed <https://arxiv.org/abs/1606.05551>`_ to use of a randomly
    drawn mutation rate. `Fast genetic algorithms <https://arxiv.org/abs/1703.03334>`_ are based on a similar idea
    These two simple methods perform quite well on a wide range of problems.

    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        *,
        noise_handling: tp.Optional[tp.Union[str, tp.Tuple[str, float]]] = None,
        mutation: str = "gaussian",
        crossover: bool = False,
        use_pareto: bool = False,
    ) -> None:
        super().__init__(_OnePlusOne, locals())


OnePlusOne = ParametrizedOnePlusOne().set_name("OnePlusOne", register=True)
NoisyOnePlusOne = ParametrizedOnePlusOne(noise_handling="random").set_name("NoisyOnePlusOne", register=True)
DiscreteOnePlusOne = ParametrizedOnePlusOne(mutation="discrete").set_name("DiscreteOnePlusOne", register=True)
PortfolioDiscreteOnePlusOne = ParametrizedOnePlusOne(mutation="portfolio").set_name(
    "PortfolioDiscreteOnePlusOne", register=True
)
DiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(mutation="lengler").set_name(
    "DiscreteLenglerOnePlusOne", register=True
)

AdaptiveDiscreteOnePlusOne = ParametrizedOnePlusOne(mutation="adaptive").set_name(
    "AdaptiveDiscreteOnePlusOne", register=True
)
AnisotropicAdaptiveDiscreteOnePlusOne = ParametrizedOnePlusOne(mutation="coordinatewise_adaptive").set_name(
    "AnisotropicAdaptiveDiscreteOnePlusOne", register=True
)

DiscreteBSOOnePlusOne = ParametrizedOnePlusOne(mutation="discreteBSO").set_name(
    "DiscreteBSOOnePlusOne", register=True
)
DiscreteDoerrOnePlusOne = ParametrizedOnePlusOne(mutation="doerr").set_name(
    "DiscreteDoerrOnePlusOne", register=True
)
DiscreteDoerrOnePlusOne.no_parallelization = True
CauchyOnePlusOne = ParametrizedOnePlusOne(mutation="cauchy").set_name("CauchyOnePlusOne", register=True)
OptimisticNoisyOnePlusOne = ParametrizedOnePlusOne(noise_handling="optimistic").set_name(
    "OptimisticNoisyOnePlusOne", register=True
)
OptimisticDiscreteOnePlusOne = ParametrizedOnePlusOne(
    noise_handling="optimistic", mutation="discrete"
).set_name("OptimisticDiscreteOnePlusOne", register=True)
NoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(
    noise_handling=("random", 1.0), mutation="discrete"
).set_name("NoisyDiscreteOnePlusOne", register=True)
DoubleFastGADiscreteOnePlusOne = ParametrizedOnePlusOne(mutation="doublefastga").set_name(
    "DoubleFastGADiscreteOnePlusOne", register=True
)
RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(
    crossover=True, mutation="portfolio", noise_handling="optimistic"
).set_name("RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne", register=True)


# pylint: too-many-arguments,too-many-instance-attributes
class _CMA(base.Optimizer):
    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        config: tp.Optional["ParametrizedCMA"] = None,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._config = ParametrizedCMA() if config is None else config
        pop = self._config.popsize
        self._popsize = max(num_workers, 4 + int(3 * np.log(self.dimension))) if pop is None else pop
        # internal attributes
        self._to_be_asked: tp.Deque[np.ndarray] = deque()
        self._to_be_told: tp.List[p.Parameter] = []
        self._num_spawners = self._popsize // 2  # experimental, for visualization
        self._parents = [self.parametrization]
        # delay initialization to ease implementation of variants
        self._es: tp.Any = None

    @property
    def es(self) -> tp.Any:  # typing not possible since cmaes not imported :(
        if self._es is None:
            if not self._config.fcmaes:
                inopts = dict(
                    popsize=self._popsize,
                    randn=self._rng.randn,
                    CMA_diagonal=self._config.diagonal,
                    verbose=-9,
                    seed=np.nan,
                    CMA_elitist=self._config.elitist,
                )
                inopts.update(self._config.inopts if self._config.inopts is not None else {})
                self._es = cma.CMAEvolutionStrategy(
                    x0=self.parametrization.sample().get_standardized_data(reference=self.parametrization)
                    if self._config.random_init
                    else np.zeros(self.dimension, dtype=np.float),
                    sigma0=self._config.scale,
                    inopts=inopts,
                )
            else:
                try:
                    from fcmaes import cmaes  # pylint: disable=import-outside-toplevel
                except ImportError as e:
                    raise ImportError(
                        "Please install fcmaes (pip install fcmaes) to use FCMA optimizers"
                    ) from e
                self._es = cmaes.Cmaes(
                    x0=np.zeros(self.dimension, dtype=np.float),
                    input_sigma=self._config.scale,
                    popsize=self._popsize,
                    randn=self._rng.randn,
                )
        return self._es

    def _internal_ask_candidate(self) -> p.Parameter:
        if not self._to_be_asked:
            self._to_be_asked.extend(self.es.ask())
        data = self._to_be_asked.popleft()
        parent = self._parents[self.num_ask % len(self._parents)]
        candidate = parent.spawn_child().set_standardized_data(data, reference=self.parametrization)
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        self._to_be_told.append(candidate)
        if len(self._to_be_told) >= self.es.popsize:
            listx = [c.get_standardized_data(reference=self.parametrization) for c in self._to_be_told]
            listy = [c.loss for c in self._to_be_told]
            args = (listy, listx) if self._config.fcmaes else (listx, listy)
            try:
                self.es.tell(*args)
            except RuntimeError:
                pass
            else:
                self._parents = sorted(self._to_be_told, key=base._loss)[: self._num_spawners]
                self._to_be_told = []

    def _internal_provide_recommendation(self) -> np.ndarray:
        pessimistic = self.current_bests["pessimistic"].parameter.get_standardized_data(
            reference=self.parametrization
        )
        if self._es is None:
            return pessimistic
        cma_best: tp.Optional[np.ndarray] = self.es.best_x if self._config.fcmaes else self.es.result.xbest
        if cma_best is None:
            return pessimistic
        return cma_best


class ParametrizedCMA(base.ConfiguredOptimizer):
    """CMA-ES optimizer,
    This evolution strategy uses a Gaussian sampling, iteratively modified
    for searching in the best directions.
    This optimizer wraps an external implementation: https://github.com/CMA-ES/pycma

    Parameters
    ----------
    scale: float
        scale of the search
    elitist: bool
        whether we switch to elitist mode, i.e. mode + instead of comma,
        i.e. mode in which we always keep the best point in the population.
    popsize: Optional[int] = None
        population size, should be n * self.num_workers for int n >= 1.
        default is max(self.num_workers, 4 + int(3 * np.log(self.dimension)))
    diagonal: bool
        use the diagonal version of CMA (advised in big dimension)
    fcmaes: bool
        use fast implementation, doesn't support diagonal=True.
        produces equivalent results, preferable for high dimensions or
        if objective function evaluation is fast.
    random_init: bool
        Use a randomized initialization
    inopts: optional dict
        use this to averride any inopts parameter of the wrapped CMA optimizer
        (see https://github.com/CMA-ES/pycma)
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        *,
        scale: float = 1.0,
        elitist: bool = False,
        popsize: tp.Optional[int] = None,
        diagonal: bool = False,
        fcmaes: bool = False,
        random_init: bool = False,
        inopts: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ) -> None:
        super().__init__(_CMA, locals(), as_config=True)
        if fcmaes:
            if diagonal:
                raise RuntimeError("fcmaes doesn't support diagonal=True, use fcmaes=False")
        self.scale = scale
        self.elitist = elitist
        self.popsize = popsize
        self.diagonal = diagonal
        self.fcmaes = fcmaes
        self.random_init = random_init
        self.inopts = inopts


CMA = ParametrizedCMA().set_name("CMA", register=True)
DiagonalCMA = ParametrizedCMA(diagonal=True).set_name("DiagonalCMA", register=True)
FCMA = ParametrizedCMA(fcmaes=True).set_name("FCMA", register=True)


class _PopulationSizeController:
    """Population control scheme for TBPSA and EDA"""

    def __init__(self, llambda: int, mu: int, dimension: int, num_workers: int = 1) -> None:
        self.llambda = max(llambda, num_workers)
        self.min_mu = min(mu, dimension)
        self.mu = mu
        self.dimension = dimension
        self.num_workers = num_workers
        self._loss_record: tp.List[float] = []

    def add_value(self, loss: tp.FloatLoss) -> None:
        self._loss_record += [loss]
        if len(self._loss_record) >= 5 * self.llambda:
            first_fifth = self._loss_record[: self.llambda]
            last_fifth = self._loss_record[-int(self.llambda) :]  # casting to int to avoid pylint bug
            means = [sum(fitnesses) / float(self.llambda) for fitnesses in [first_fifth, last_fifth]]
            stds = [np.std(fitnesses) / np.sqrt(self.llambda - 1) for fitnesses in [first_fifth, last_fifth]]
            z = (means[0] - means[1]) / (np.sqrt(stds[0] ** 2 + stds[1] ** 2))
            if z < 2.0:
                self.mu *= 2
            else:
                self.mu = max(self.min_mu, int(self.mu * 0.84))
            self.llambda = 4 * self.mu
            if self.num_workers > 1:
                self.llambda = max(self.llambda, self.num_workers)
                self.mu = self.llambda // 4
            self._loss_record = []


# pylint: disable=too-many-instance-attributes
@registry.register
class EDA(base.Optimizer):
    """Test-based population-size adaptation.

    Population-size equal to lambda = 4 x dimension.
    Test by comparing the first fifth and the last fifth of the 5lambda evaluations.

    Caution
    -------
    This optimizer is probably wrong.
    """

    _POPSIZE_ADAPTATION = False
    _COVARIANCE_MEMORY = False

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.sigma = 1
        self.covariance = np.identity(self.dimension)
        dim = self.dimension
        self.popsize = _PopulationSizeController(
            llambda=4 * dim, mu=dim, dimension=dim, num_workers=num_workers
        )
        self.current_center: np.ndarray = np.zeros(self.dimension)
        # Population
        self.children: tp.List[p.Parameter] = []
        self.parents: tp.List[p.Parameter] = [
            self.parametrization
        ]  # for transfering heritage (checkpoints in PBT)

    def _internal_provide_recommendation(
        self,
    ) -> tp.ArrayLike:  # This is NOT the naive version. We deal with noise.
        return self.current_center

    def _internal_ask_candidate(self) -> p.Parameter:
        mutated_sigma = self.sigma * np.exp(self._rng.normal(0, 1) / np.sqrt(self.dimension))
        # TODO: is a sigma necessary here as well? given the covariance is estimated
        assert len(self.current_center) == len(self.covariance), [
            self.dimension,
            self.current_center,
            self.covariance,
        ]
        data = self._rng.multivariate_normal(self.current_center, mutated_sigma * self.covariance)
        parent = self.parents[self.num_ask % len(self.parents)]
        candidate = parent.spawn_child().set_standardized_data(data, reference=self.parametrization)
        if parent is self.parametrization:
            candidate.heritage["lineage"] = candidate.uid  # for tracking
        candidate._meta["sigma"] = mutated_sigma
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        self.children.append(candidate)
        if self._POPSIZE_ADAPTATION:
            self.popsize.add_value(loss)
        if len(self.children) >= self.popsize.llambda:
            self.children = sorted(self.children, key=base._loss)
            population_data = [c.get_standardized_data(reference=self.parametrization) for c in self.children]
            mu = self.popsize.mu
            arrays = population_data[:mu]
            # covariance
            # TODO: check actual covariance that should be used
            centered_arrays = np.array([x - self.current_center for x in arrays])
            cov = centered_arrays.T.dot(centered_arrays)
            # cov = np.cov(np.array(population_data).T)
            mem_factor = 0.9 if self._COVARIANCE_MEMORY else 0
            self.covariance *= mem_factor
            self.covariance += (1 - mem_factor) * cov
            # Computing the new parent
            self.current_center = sum(arrays) / mu  # type: ignore
            self.sigma = np.exp(sum([np.log(c._meta["sigma"]) for c in self.children[:mu]]) / mu)
            self.parents = self.children[:mu]
            self.children = []

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        raise errors.TellNotAskedNotSupportedError


class PCEDA(EDA):
    _POPSIZE_ADAPTATION = True
    _COVARIANCE_MEMORY = False


class MPCEDA(EDA):
    _POPSIZE_ADAPTATION = True
    _COVARIANCE_MEMORY = True


class MEDA(EDA):
    _POPSIZE_ADAPTATION = False
    _COVARIANCE_MEMORY = True


class _TBPSA(base.Optimizer):
    """Test-based population-size adaptation.

    Population-size equal to lambda = 4 x dimension.
    Test by comparing the first fifth and the last fifth of the 5lambda evaluations.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        naive: bool = True,
        initial_popsize: tp.Optional[int] = None,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.sigma = 1
        self.naive = naive
        if initial_popsize is None:
            initial_popsize = self.dimension
        self.popsize = _PopulationSizeController(
            llambda=4 * initial_popsize, mu=initial_popsize, dimension=self.dimension, num_workers=num_workers
        )
        self.current_center: np.ndarray = np.zeros(self.dimension)
        # population
        self.parents: tp.List[p.Parameter] = [
            self.parametrization
        ]  # for transfering heritage (checkpoints in PBT)
        self.children: tp.List[p.Parameter] = []

    def recommend(self) -> p.Parameter:
        if self.naive:
            return self.current_bests["optimistic"].parameter
        else:
            # This is NOT the naive version. We deal with noise.
            out = self.parametrization.spawn_child()
            with p.helpers.deterministic_sampling(out):
                out.set_standardized_data(self.current_center)
            return out

    def _internal_ask_candidate(self) -> p.Parameter:
        mutated_sigma = self.sigma * np.exp(self._rng.normal(0, 1) / np.sqrt(self.dimension))
        individual = self.current_center + mutated_sigma * self._rng.normal(0, 1, self.dimension)
        parent = self.parents[self.num_ask % len(self.parents)]
        candidate = parent.spawn_child().set_standardized_data(individual, reference=self.parametrization)
        if parent is self.parametrization:
            candidate.heritage["lineage"] = candidate.uid  # for tracking
        candidate._meta["sigma"] = mutated_sigma
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        self.popsize.add_value(loss)
        self.children.append(candidate)
        if len(self.children) >= self.popsize.llambda:
            # Sorting the population.
            self.children.sort(key=base._loss)
            # Computing the new parent.

            self.parents = self.children[: self.popsize.mu]
            self.children = []
            self.current_center = (
                sum(  # type: ignore
                    c.get_standardized_data(reference=self.parametrization) for c in self.parents
                )
                / self.popsize.mu
            )
            self.sigma = np.exp(np.sum(np.log([c._meta["sigma"] for c in self.parents])) / self.popsize.mu)

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        data = candidate.get_standardized_data(reference=self.parametrization)
        sigma = np.linalg.norm(data - self.current_center) / np.sqrt(self.dimension)  # educated guess
        candidate._meta["sigma"] = sigma
        self._internal_tell_candidate(candidate, loss)  # go through standard pipeline


class ParametrizedTBPSA(base.ConfiguredOptimizer):
    """`Test-based population-size adaptation <https://homepages.fhv.at/hgb/New-Papers/PPSN16_HB16.pdf>`_
    This method, based on adapting the population size, performs the best in
    many noisy optimization problems, even in large dimension

    Parameters
    ----------
    naive: bool
        set to False for noisy problem, so that the best points will be an
        average of the final population.
    initial_popsize: Optional[int]
        initial (and minimal) population size (default: 4 x dimension)

    Note
    ----
    Derived from:
    Hellwig, Michael & Beyer, Hans-Georg. (2016).
    Evolution under Strong Noise: A Self-Adaptive Evolution Strategy
    Reaches the Lower Performance Bound -- the pcCMSA-ES.
    https://homepages.fhv.at/hgb/New-Papers/PPSN16_HB16.pdf
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        *,
        naive: bool = True,
        initial_popsize: tp.Optional[int] = None,
    ) -> None:
        super().__init__(_TBPSA, locals())


TBPSA = ParametrizedTBPSA(naive=False).set_name("TBPSA", register=True)
NaiveTBPSA = ParametrizedTBPSA().set_name("NaiveTBPSA", register=True)


@registry.register
class NoisyBandit(base.Optimizer):
    """UCB.
    This is upper confidence bound (adapted to minimization),
    with very poor parametrization; in particular, the logarithmic term is set to zero.
    Infinite arms: we add one arm when `20 * #ask >= #arms ** 3`.
    """

    def _internal_ask(self) -> tp.ArrayLike:
        if 20 * self._num_ask >= len(self.archive) ** 3:
            return self._rng.normal(0, 1, self.dimension)  # type: ignore
        if self._rng.choice([True, False]):
            # numpy does not accept choice on list of tuples, must choose index instead
            idx = self._rng.choice(len(self.archive))
            return np.frombuffer(list(self.archive.bytesdict.keys())[idx])  # type: ignore
        return self.current_bests["optimistic"].x


class _PSO(base.Optimizer):

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        config: tp.Optional["ConfiguredPSO"] = None,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._config = ConfiguredPSO() if config is None else config
        if budget is not None and budget < 60:
            warnings.warn("PSO is inefficient with budget < 60", errors.InefficientSettingsWarning)
        cases: tp.Dict[str, tp.Tuple[tp.Optional[float], transforms.Transform]] = dict(
            arctan=(0, transforms.ArctanBound(0, 1)),
            identity=(None, transforms.Affine(1, 0)),
            gaussian=(1e-10, transforms.CumulativeDensity()),
        )
        # eps is used for clipping to make sure it is admissible
        self._eps, self._transform = cases[self._config.transform]
        self.llambda = max(40, num_workers)
        if self._config.popsize is not None:
            self.llambda = self._config.popsize
        self._uid_queue = base.utils.UidQueue()
        self.population: tp.Dict[str, p.Parameter] = {}
        self._best = self.parametrization.spawn_child()

    def _internal_ask_candidate(self) -> p.Parameter:
        # population is increased only if queue is empty (otherwise tell_not_asked does not work well at the beginning)
        if len(self.population) < self.llambda:
            candidate = self.parametrization.sample()
            self.population[candidate.uid] = candidate
            dim = self.parametrization.dimension
            candidate.heritage["speed"] = (
                self._rng.normal(size=dim) if self._eps is None else self._rng.uniform(-1, 1, dim)
            )
            self._uid_queue.asked.add(candidate.uid)
            return candidate
        uid = self._uid_queue.ask()
        candidate = self._spawn_mutated_particle(self.population[uid])
        candidate.heritage["lineage"] = uid  # override in case it was a tell-not-asked
        return candidate

    def _get_boxed_data(self, particle: p.Parameter) -> np.ndarray:
        if particle._frozen and "boxed_data" in particle._meta:
            return particle._meta["boxed_data"]  # type: ignore
        boxed_data = self._transform.forward(particle.get_standardized_data(reference=self.parametrization))
        if particle._frozen:  # only save is frozen
            particle._meta["boxed_data"] = boxed_data
        return boxed_data

    def _spawn_mutated_particle(self, particle: p.Parameter) -> p.Parameter:
        x = self._get_boxed_data(particle)
        speed: np.ndarray = particle.heritage["speed"]
        global_best_x = self._get_boxed_data(self._best)
        parent_best_x = self._get_boxed_data(particle.heritage.get("best_parent", particle))
        rp = self._rng.uniform(0.0, 1.0, size=self.dimension)
        rg = self._rng.uniform(0.0, 1.0, size=self.dimension)
        speed = (
            self._config.omega * speed
            + self._config.phip * rp * (parent_best_x - x)
            + self._config.phig * rg * (global_best_x - x)
        )
        data = speed + x
        if self._eps is not None:
            data = np.clip(data, self._eps, 1 - self._eps)
        data = self._transform.backward(data)
        new_part = particle.spawn_child().set_standardized_data(data, reference=self.parametrization)
        new_part.heritage["speed"] = speed
        return new_part

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        uid = candidate.heritage["lineage"]
        if uid not in self.population:
            self._internal_tell_not_asked(candidate, loss)
            return
        self._uid_queue.tell(uid)
        self.population[uid] = candidate
        if self._best.loss is None or loss < self._best.loss:
            self._best = candidate
        if loss <= candidate.heritage.get("best_parent", candidate).loss:
            candidate.heritage["best_parent"] = candidate

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        # nearly same as DE
        worst: tp.Optional[p.Parameter] = None
        if not len(self.population) < self.llambda:
            uid, worst = max(self.population.items(), key=lambda p: base._loss(p[1]))
            if base._loss(worst) < loss:
                return  # no need to update
            else:
                del self.population[uid]
                self._uid_queue.discard(uid)
        if "speed" not in candidate.heritage:
            candidate.heritage["speed"] = self._rng.uniform(-1.0, 1.0, self.parametrization.dimension)
        self.population[candidate.uid] = candidate
        self._uid_queue.tell(candidate.uid)
        if loss < base._loss(self._best):
            self._best = candidate


class ConfiguredPSO(base.ConfiguredOptimizer):
    """`Particle Swarm Optimization <https://en.wikipedia.org/wiki/Particle_swarm_optimization>`_
    is based on a set of particles with their inertia.
    Wikipedia provides a beautiful illustration ;) (see link)


    Parameters
    ----------
    transform: str
        name of the transform to use to map from PSO optimization space to R-space.
    popsize: int
        population size of the particle swarm. Defaults to max(40, num_workers)
    omega: float
        particle swarm optimization parameter
    phip: float
        particle swarm optimization parameter
    phig: float
        particle swarm optimization parameter

    Note
    ----
    - Using non-default "transform" and "wide" parameters can lead to extreme values
    - Implementation partially following SPSO2011. However, no randomization of the population order.
    - Reference:
      M. Zambrano-Bigiarini, M. Clerc and R. Rojas,
      Standard Particle Swarm Optimisation 2011 at CEC-2013: A baseline for future PSO improvements,
      2013 IEEE Congress on Evolutionary Computation, Cancun, 2013, pp. 2337-2344.
      https://ieeexplore.ieee.org/document/6557848
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        transform: str = "identity",
        popsize: tp.Optional[int] = None,
        omega: float = 0.5 / math.log(2.0),
        phip: float = 0.5 + math.log(2.0),
        phig: float = 0.5 + math.log(2.0),
    ) -> None:
        super().__init__(_PSO, locals(), as_config=True)
        assert transform in ["arctan", "gaussian", "identity"]
        self.transform = transform
        self.popsize = popsize
        self.omega = omega
        self.phip = phip
        self.phig = phig


RealSpacePSO = ConfiguredPSO().set_name("RealSpacePSO", register=True)
PSO = ConfiguredPSO(transform="arctan").set_name("PSO", register=True)


@registry.register
class SPSA(base.Optimizer):
    # pylint: disable=too-many-instance-attributes
    """The First order SPSA algorithm as shown in [1,2,3], with implementation details
    from [4,5].

    1) https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation
    2) https://www.chessprogramming.org/SPSA
    3) Spall, James C. "Multivariate stochastic approximation using a simultaneous perturbation gradient approximation."
       IEEE transactions on automatic control 37.3 (1992): 332-341.
    4) Section 7.5.2 in "Introduction to Stochastic Search and Optimization: Estimation, Simulation and Control" by James C. Spall.
    5) Pushpendre Rastogi, Jingyi Zhu, James C. Spall CISS (2016).
       Efficient implementation of Enhanced Adaptive Simultaneous Perturbation Algorithms.
    """
    no_parallelization = True

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.init = True
        self.idx = 0
        self.delta = float("nan")
        self.ym: tp.Optional[np.ndarray] = None
        self.yp: tp.Optional[np.ndarray] = None
        self.t: np.ndarray = np.zeros(self.dimension)
        self.avg: np.ndarray = np.zeros(self.dimension)
        # Set A, a, c according to the practical implementation
        # guidelines in the ISSO book.
        self.A = 10 if budget is None else max(10, budget // 20)
        # TODO: We should spend first 10-20 iterations
        # to estimate the noise standard deviation and
        # then set c = standard deviation. 1e-1 is arbitrary.
        self.c = 1e-1
        # TODO: We should chose a to be inversely proportional to
        # the magnitude of gradient and proportional to (1+A)^0.602
        # we should spend some burn-in iterations to estimate the
        # magnitude of the gradient. 1e-5 is arbitrary.
        self.a = 1e-5

    def _ck(self, k: int) -> float:
        "c_k determines the pertubation."
        return self.c / (k // 2 + 1) ** 0.101

    def _ak(self, k: int) -> float:
        "a_k is the learning rate."
        return self.a / (k // 2 + 1 + self.A) ** 0.602

    def _internal_ask(self) -> tp.ArrayLike:
        k = self.idx
        if k % 2 == 0:
            if not self.init:
                assert self.yp is not None and self.ym is not None
                self.t -= (self._ak(k) * (self.yp - self.ym) / 2 / self._ck(k)) * self.delta
                self.avg += (self.t - self.avg) / (k // 2 + 1)
            self.delta = 2 * self._rng.randint(2, size=self.dimension) - 1
            return self.t - self._ck(k) * self.delta  # type:ignore
        return self.t + self._ck(k) * self.delta  # type: ignore

    def _internal_tell(self, x: tp.ArrayLike, loss: tp.FloatLoss) -> None:
        setattr(self, ("ym" if self.idx % 2 == 0 else "yp"), np.array(loss, copy=True))
        self.idx += 1
        if self.init and self.yp is not None and self.ym is not None:
            self.init = False

    def _internal_provide_recommendation(self) -> tp.ArrayLike:
        return self.avg


class _Rescaled(base.Optimizer):
    """Proposes a version of a base optimizer which works at a different scale."""

    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        base_optimizer: base.OptCls = CMA,
        scale: tp.Optional[float] = None,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._optimizer = base_optimizer(self.parametrization, budget=budget, num_workers=num_workers)
        self._subcandidates: tp.Dict[str, p.Parameter] = {}
        if scale is None:
            assert budget is not None, "Either scale or budget must be known in _Rescaled."
            scale = np.sqrt(np.log(self.budget) / self.dimension)
        self.scale = scale
        assert self.scale != 0.0, "scale should be non-zero in Rescaler."

    def rescale_candidate(self, candidate: p.Parameter, inverse: bool = False) -> p.Parameter:
        data = candidate.get_standardized_data(reference=self.parametrization)
        scale = self.scale if not inverse else 1.0 / self.scale
        return self.parametrization.spawn_child().set_standardized_data(scale * data)

    def _internal_ask_candidate(self) -> p.Parameter:
        candidate = self._optimizer.ask()
        sent_candidate = self.rescale_candidate(candidate)
        # We store the version corresponding to the underlying optimizer.
        self._subcandidates[sent_candidate.uid] = candidate
        return sent_candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        self._optimizer.tell(self._subcandidates.pop(candidate.uid), loss)

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        candidate = self.rescale_candidate(candidate, inverse=True)
        self._optimizer.tell(candidate, loss)


class SplitOptimizer(base.Optimizer):
    """Combines optimizers, each of them working on their own variables.

    Parameters
    ---------
    num_optims: int or None
        number of optimizers
    num_vars: int or None
        number of variable per optimizer.
    progressive: bool
        True if we want to progressively add optimizers during the optimization run.
        If progressive = True, the optimizer is forced at OptimisticNoisyOnePlusOne.

    Example
    -------
    for 5 optimizers, each of them working on 2 variables, one can use:

    opt = SplitOptimizer(parametrization=10, num_workers=3, num_optims=5, num_vars=[2, 2, 2, 2, 2])
    or equivalently:
    opt = SplitOptimizer(parametrization=10, num_workers=3, num_vars=[2, 2, 2, 2, 2])
    Given that all optimizers have the same number of variables, one can also run:
    opt = SplitOptimizer(parametrization=10, num_workers=3, num_optims=5)

    Note
    ----
    By default, it uses CMA for multivariate groups and RandomSearch for monovariate groups.

    Caution
    -------
    The variables refer to the deep representation used by optimizers.
    For example, a categorical variable with 5 possible values becomes 5 continuous variables.
    """

    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        num_optims: tp.Optional[int] = None,
        num_vars: tp.Optional[tp.List[int]] = None,
        multivariate_optimizer: base.OptCls = CMA,
        monovariate_optimizer: base.OptCls = OnePlusOne,
        progressive: bool = False,
        non_deterministic_descriptor: bool = True,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._subcandidates: tp.Dict[str, tp.List[p.Parameter]] = {}
        self._progressive = progressive
        subparams: tp.List[p.Parameter] = []
        if num_vars is not None:  # The user has specified how are the splits (s)he wants.
            assert (
                sum(num_vars) == self.dimension
            ), f"sum(num_vars)={sum(num_vars)} should be equal to the dimension {self.dimension}."
            if num_optims is None:  # we deduce the number of splits.
                num_optims = len(num_vars)
            assert num_optims == len(
                num_vars
            ), f"The number {num_optims} of optimizers should match len(num_vars)={len(num_vars)}."
        elif num_optims is None:
            # if no num_vars and no num_optims, try to guess how to split. Otherwise, just assume 2.
            if isinstance(parametrization, p.Parameter):
                subparams = p.helpers.list_data(parametrization)  # type: ignore
                if len(subparams) == 1:
                    subparams.clear()
                num_optims = len(subparams)
            if not subparams:  # Desperate situation: just split in 2.
                num_optims = 2
        if not subparams:
            # if num_vars not given: we will distribute variables equally.
            assert num_optims is not None
            num_optims = min(num_optims, self.dimension)
            num_vars = num_vars if num_vars else []
            for i in range(num_optims):
                if len(num_vars) < i + 1:
                    num_vars += [(self.dimension // num_optims) + (self.dimension % num_optims > i)]
                assert num_vars[i] >= 1, "At least one variable per optimizer."
                subparams += [p.Array(shape=(num_vars[i],))]
        if non_deterministic_descriptor:
            for param in subparams:
                param.function.deterministic = False
        # synchronize random state and create optimizers
        self.optims: tp.List[base.Optimizer] = []
        mono, multi = monovariate_optimizer, multivariate_optimizer
        for param in subparams:
            param.random_state = self.parametrization.random_state
            self.optims.append((multi if param.dimension > 1 else mono)(param, budget, num_workers))
        # final check for dimension
        assert (
            sum(opt.dimension for opt in self.optims) == self.dimension
        ), "sum of sub-dimensions should be equal to the total dimension."

    def _internal_ask_candidate(self) -> p.Parameter:
        candidates: tp.List[p.Parameter] = []
        for i, opt in enumerate(self.optims):
            if self._progressive:
                assert self.budget is not None
                if i > 0 and i / len(self.optims) > np.sqrt(2.0 * self.num_ask / self.budget):
                    candidates.append(opt.parametrization.spawn_child())  # unchanged
                    continue
            candidates.append(opt.ask())
        data = np.concatenate(
            [
                c.get_standardized_data(reference=opt.parametrization)
                for c, opt in zip(candidates, self.optims)
            ],
            axis=0,
        )
        cand = self.parametrization.spawn_child().set_standardized_data(data)
        self._subcandidates[cand.uid] = candidates
        return cand

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        candidates = self._subcandidates.pop(candidate.uid)
        for cand, opt in zip(candidates, self.optims):
            opt.tell(cand, loss)

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        data = candidate.get_standardized_data(reference=self.parametrization)
        start = 0
        for opt in self.optims:
            local_data = data[start : start + opt.dimension]
            start += opt.dimension
            local_candidate = opt.parametrization.spawn_child().set_standardized_data(local_data)
            opt.tell(local_candidate, loss)


class Rescaled(base.ConfiguredOptimizer):
    """Configured optimizer for creating rescaled optimization algorithms.

    Parameters
    ----------
    base_optimizer: base.OptCls
        optimization algorithm to be rescaled.
    scale: how much do we rescale. E.g. 0.001 if we want to focus on the center
        with std 0.001 (assuming the std of the domain is set to 1).
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        *,
        base_optimizer: base.OptCls = CMA,
        scale: tp.Optional[float] = None,
    ) -> None:
        super().__init__(_Rescaled, locals())


RescaledCMA = Rescaled().set_name("RescaledCMA", register=True)


class ConfSplitOptimizer(base.ConfiguredOptimizer):
    """ "Combines optimizers, each of them working on their own variables.

    Parameters
    ----------
    num_optims: int
        number of optimizers
    num_vars: optional list of int
        number of variable per optimizer.
    progressive: optional bool
        whether we progressively add optimizers.
    non_deterministic_descriptor: bool
        subparts parametrization descriptor is set to noisy function.
        This can have an impact for optimizer selection for competence maps.
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        *,
        num_optims: tp.Optional[int] = None,
        num_vars: tp.Optional[tp.List[int]] = None,
        multivariate_optimizer: base.OptCls = CMA,
        monovariate_optimizer: base.OptCls = RandomSearch,
        progressive: bool = False,
        non_deterministic_descriptor: bool = True,
    ) -> None:
        super().__init__(SplitOptimizer, locals())


@registry.register
class Portfolio(base.Optimizer):
    """Passive portfolio of CMA, 2-pt DE and Scr-Hammersley."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        assert budget is not None
        self.optims = [
            CMA(
                self.parametrization, budget // 3 + (budget % 3 > 0), num_workers
            ),  # share parametrization and its rng
            TwoPointsDE(self.parametrization, budget // 3 + (budget % 3 > 1), num_workers),  # noqa: F405
            ScrHammersleySearch(self.parametrization, budget // 3, num_workers),
        ]  # noqa: F405
        if budget < 12 * num_workers:
            self.optims = [ScrHammersleySearch(self.parametrization, budget, num_workers)]  # noqa: F405

    def _internal_ask_candidate(self) -> p.Parameter:
        optim_index = self._num_ask % len(self.optims)
        candidate = self.optims[optim_index].ask()
        candidate._meta["optim_index"] = optim_index
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        for opt in self.optims:
            try:
                opt.tell(candidate, loss)
            except errors.TellNotAskedNotSupportedError:
                pass
        # Presumably better than self.optims[optim_index].tell(candidate, value)

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        at_least_one_ok = False
        for opt in self.optims:
            try:
                opt.tell(candidate, loss)
                at_least_one_ok = True
            except errors.TellNotAskedNotSupportedError:
                pass
        if not at_least_one_ok:
            raise errors.TellNotAskedNotSupportedError


class InfiniteMetaModelOptimum(ValueError):
    """Sometimes the optimum of the metamodel is at infinity."""


def learn_on_k_best(archive: utils.Archive[utils.MultiValue], k: int) -> tp.ArrayLike:
    """Approximate optimum learnt from the k best.

    Parameters
    ----------
    archive: utils.Archive[utils.Value]
    """
    items = list(archive.items_as_arrays())
    dimension = len(items[0][0])

    # Select the k best.
    first_k_individuals = [
        x for x in sorted(items, key=lambda indiv: archive[indiv[0]].get_estimation("pessimistic"))[:k]
    ]
    assert len(first_k_individuals) == k

    # Recenter the best.
    middle = np.array(sum(p[0] for p in first_k_individuals) / k)
    normalization = 1e-15 + np.sqrt(np.sum((first_k_individuals[-1][0] - first_k_individuals[0][0]) ** 2))
    y = [archive[c[0]].get_estimation("pessimistic") for c in first_k_individuals]
    X = np.asarray([(c[0] - middle) / normalization for c in first_k_individuals])

    # We need SKLearn.
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    polynomial_features = PolynomialFeatures(degree=2)
    X2 = polynomial_features.fit_transform(X)

    # Fit a linear model.
    model = LinearRegression()
    model.fit(X2, y)

    try:
        for cls in Powell, DE:  # Powell excellent here, DE as a backup for thread safety.
            optimizer = cls(parametrization=dimension, budget=45 * dimension + 30)
            try:
                minimum = optimizer.minimize(
                    lambda x: float(model.predict(polynomial_features.fit_transform(x[None, :])))
                ).value
            except RuntimeError:
                assert cls == Powell, "Only Powell is allowed to crash here."
            else:
                break
    except ValueError:
        raise InfiniteMetaModelOptimum("Infinite meta-model optimum in learn_on_k_best.")

    if np.sum(minimum ** 2) > 1.0:
        raise InfiniteMetaModelOptimum("huge meta-model optimum in learn_on_k_best.")
    return middle + normalization * minimum


@registry.register
class MetaModel(base.Optimizer):
    """Adding a metamodel into CMA."""

    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        multivariate_optimizer: tp.Optional[base.OptCls] = None,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        if multivariate_optimizer is None:
            multivariate_optimizer = CMA if self.dimension > 1 else OnePlusOne
        self._optim = multivariate_optimizer(
            self.parametrization, budget, num_workers
        )  # share parametrization and its rng

    def _internal_ask_candidate(self) -> p.Parameter:
        # We request a bit more points than what is really necessary for our dimensionality (+dimension).
        sample_size = int((self.dimension * (self.dimension - 1)) / 2 + 2 * self.dimension + 1)
        if (
            self._num_ask % max(13, self.num_workers, self.dimension) == 0
            and len(self.archive) >= sample_size
        ):
            try:
                data = learn_on_k_best(self.archive, sample_size)
                candidate = self.parametrization.spawn_child().set_standardized_data(data)
            except InfiniteMetaModelOptimum:  # The optimum is at infinity. Shit happens.
                candidate = self._optim.ask()
        else:
            candidate = self._optim.ask()
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        self._optim.tell(candidate, loss)


class ParaPortfolio(Portfolio):
    """Passive portfolio of CMA, 2-pt DE, PSO, SQP and Scr-Hammersley."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        assert budget is not None

        def intshare(n: int, m: int) -> tp.Tuple[int, ...]:
            x = [n // m] * m
            i = 0
            while sum(x) < n:
                x[i] += 1
                i += 1
            return tuple(x)

        nw1, nw2, nw3, nw4 = intshare(num_workers - 1, 4)
        self.which_optim = [0] * nw1 + [1] * nw2 + [2] * nw3 + [3] + [4] * nw4
        assert len(self.which_optim) == num_workers
        # b1, b2, b3, b4, b5 = intshare(budget, 5)
        self.optims: tp.List[base.Optimizer] = [
            CMA(self.parametrization, num_workers=nw1),  # share parametrization and its rng
            TwoPointsDE(self.parametrization, num_workers=nw2),  # noqa: F405
            PSO(self.parametrization, num_workers=nw3),
            SQP(self.parametrization, num_workers=1),  # noqa: F405
            ScrHammersleySearch(
                self.parametrization, budget=(budget // len(self.which_optim)) * nw4
            ),  # noqa: F405
        ]

    def _internal_ask_candidate(self) -> p.Parameter:
        optim_index = self.which_optim[self._num_ask % len(self.which_optim)]
        candidate = self.optims[optim_index].ask()
        candidate._meta["optim_index"] = optim_index
        return candidate


@registry.register
class SQPCMA(ParaPortfolio):
    """Passive portfolio of CMA and many SQP."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        assert budget is not None
        nw = num_workers // 2
        self.which_optim = [0] * nw
        for i in range(num_workers - nw):
            self.which_optim += [i + 1]
        assert len(self.which_optim) == num_workers
        # b1, b2, b3, b4, b5 = intshare(budget, 5)
        self.optims = [CMA(self.parametrization, num_workers=nw)]  # share parametrization and its rng
        for i in range(num_workers - nw):
            self.optims += [SQP(self.parametrization, num_workers=1)]  # noqa: F405
            if i > 0:
                self.optims[-1].initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore


@registry.register
class ASCMADEthird(Portfolio):
    """Algorithm selection, with CMA and Lhs-DE. Active selection at 1/3."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        assert budget is not None
        self.optims = [
            CMA(
                self.parametrization, budget=None, num_workers=num_workers
            ),  # share parametrization and its rng
            LhsDE(self.parametrization, budget=None, num_workers=num_workers),
        ]  # noqa: F405
        self.budget_before_choosing = budget // 3
        self.best_optim = -1

    def _internal_ask_candidate(self) -> p.Parameter:
        if self.budget_before_choosing > 0:
            self.budget_before_choosing -= 1
            optim_index = self._num_ask % len(self.optims)
        else:
            if self.best_optim is None:
                best_loss = float("inf")
                optim_index = -1
                for i, optim in enumerate(self.optims):
                    val = optim.current_bests["pessimistic"].get_estimation("pessimistic")
                    if not val > best_loss:
                        optim_index = i
                        best_loss = val
                self.best_optim = optim_index
            optim_index = self.best_optim
        candidate = self.optims[optim_index].ask()
        candidate._meta["optim_index"] = optim_index
        return candidate


@registry.register
class CMandAS2(ASCMADEthird):
    """Competence map, with algorithm selection in one of the cases (3 CMAs)."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.optims = [TwoPointsDE(self.parametrization, budget=None, num_workers=num_workers)]  # noqa: F405
        assert budget is not None
        self.budget_before_choosing = 2 * budget
        if budget < 201:
            self.optims = [OnePlusOne(self.parametrization, budget=None, num_workers=num_workers)]
        if budget > 50 * self.dimension or num_workers < 30:
            self.optims = [
                MetaModel(self.parametrization, budget=None, num_workers=num_workers) for _ in range(3)
            ]
            self.budget_before_choosing = budget // 10


@registry.register
class CMandAS3(ASCMADEthird):
    """Competence map, with algorithm selection in one of the cases (3 CMAs)."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.optims = [TwoPointsDE(self.parametrization, budget=None, num_workers=num_workers)]  # noqa: F405
        assert budget is not None
        self.budget_before_choosing = 2 * budget
        if budget < 201:
            self.optims = [OnePlusOne(self.parametrization, budget=None, num_workers=num_workers)]
        if budget > 50 * self.dimension or num_workers < 30:
            if num_workers == 1:
                self.optims = [
                    ChainCMAPowell(
                        self.parametrization, budget=None, num_workers=num_workers
                    ),  # share parametrization and its rng
                    ChainCMAPowell(self.parametrization, budget=None, num_workers=num_workers),
                    ChainCMAPowell(self.parametrization, budget=None, num_workers=num_workers),
                ]
            else:
                self.optims = [
                    CMA(
                        self.parametrization, budget=None, num_workers=num_workers
                    ),  # share parametrization and its rng
                    CMA(self.parametrization, budget=None, num_workers=num_workers),
                    CMA(self.parametrization, budget=None, num_workers=num_workers),
                ]
            self.budget_before_choosing = budget // 10


@registry.register
class CM(CMandAS2):
    """Competence map, simplest."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        assert budget is not None
        # share parametrization and its random number generator between all underlying optimizers
        self.optims = [TwoPointsDE(self.parametrization, budget=None, num_workers=num_workers)]  # noqa: F405
        self.budget_before_choosing = 2 * budget
        if budget < 201:
            self.optims = [OnePlusOne(self.parametrization, budget=None, num_workers=num_workers)]
        if budget > 50 * self.dimension:
            self.optims = [CMA(self.parametrization, budget=None, num_workers=num_workers)]


@registry.register
class MultiCMA(CM):
    """Combining 3 CMAs. Exactly identical. Active selection at 1/10 of the budget."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        assert budget is not None
        self.optims = [
            CMA(
                self.parametrization, budget=None, num_workers=num_workers
            ),  # share parametrization and its rng
            CMA(self.parametrization, budget=None, num_workers=num_workers),
            CMA(self.parametrization, budget=None, num_workers=num_workers),
        ]
        self.budget_before_choosing = budget // 10


@registry.register
class MultiDiscrete(CM):
    """Combining 3 Discrete(1+1). Exactly identical. Active selection at 1/10 of the budget."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        assert budget is not None
        self.optims = [
            DiscreteOnePlusOne(
                self.parametrization, budget=budget // 12, num_workers=num_workers
            ),  # share parametrization and its rng
            DiscreteBSOOnePlusOne(self.parametrization, budget=budget // 12, num_workers=num_workers),
            DoubleFastGADiscreteOnePlusOne(
                self.parametrization, budget=(budget // 4) - 2 * (budget // 12), num_workers=num_workers
            ),
        ]
        self.budget_before_choosing = budget // 4


@registry.register
class TripleCMA(CM):
    """Combining 3 CMAs. Exactly identical. Active selection at 1/3 of the budget."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        assert budget is not None
        self.optims = [
            ParametrizedCMA(random_init=True)(
                self.parametrization, budget=None, num_workers=num_workers
            ),  # share parametrization and its rng
            ParametrizedCMA(random_init=True)(self.parametrization, budget=None, num_workers=num_workers),
            ParametrizedCMA(random_init=True)(self.parametrization, budget=None, num_workers=num_workers),
        ]
        self.budget_before_choosing = budget // 3


@registry.register
class PolyCMA(CM):
    """Combining 20 CMAs. Exactly identical. Active selection at 1/3 of the budget."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        assert budget is not None
        self.optims = [
            ParametrizedCMA(random_init=True)(self.parametrization, budget=None, num_workers=num_workers)
            for _ in range(20)
        ]

        self.budget_before_choosing = budget // 3


@registry.register
class MultiScaleCMA(CM):
    """Combining 3 CMAs with different init scale. Active selection at 1/3 of the budget."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.optims = [
            CMA(
                self.parametrization, budget=None, num_workers=num_workers
            ),  # share parametrization and its rng
            ParametrizedCMA(scale=1e-3, random_init=True)(
                self.parametrization, budget=None, num_workers=num_workers
            ),
            ParametrizedCMA(scale=1e-6, random_init=True)(
                self.parametrization, budget=None, num_workers=num_workers
            ),
        ]
        assert budget is not None
        self.budget_before_choosing = budget // 3


class _FakeFunction:
    """Simple function that returns the loss which was registered just before.
    This is a hack for BO.
    """

    def __init__(self, num_digits: int) -> None:
        self.num_digits = num_digits
        self._registered: tp.List[tp.Tuple[np.ndarray, float]] = []

    def key(self, num: int) -> str:
        """Key corresponding to the array sample
        (uses zero-filling to keep order)
        """
        return "x" + str(num).zfill(self.num_digits)

    def register(self, x: np.ndarray, loss: tp.FloatLoss) -> None:
        if self._registered:
            raise RuntimeError("Only one call can be registered at a time")
        self._registered.append((x, loss))

    def __call__(self, **kwargs: float) -> float:
        if not self._registered:
            raise RuntimeError("Call must be registered first")
        x = [kwargs[self.key(i)] for i in range(len(kwargs))]
        xr, loss = self._registered[0]
        np.testing.assert_array_almost_equal(x, xr, err_msg="Call does not match registered")
        self._registered.clear()
        return loss


class _BO(base.Optimizer):
    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        *,
        initialization: tp.Optional[str] = None,
        init_budget: tp.Optional[int] = None,
        middle_point: bool = False,
        utility_kind: str = "ucb",  # bayes_opt default
        utility_kappa: float = 2.576,
        utility_xi: float = 0.0,
        gp_parameters: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._transform = transforms.ArctanBound(0, 1)
        self._bo: tp.Optional[BayesianOptimization] = None
        self._fake_function = _FakeFunction(num_digits=len(str(self.dimension)))
        # initialization
        init = initialization
        self._init_budget = init_budget
        self._middle_point = middle_point
        if init is None:
            self._InitOpt: tp.Optional[base.ConfiguredOptimizer] = None
        elif init == "random":
            self._InitOpt = oneshot.RandomSearch
        else:
            self._InitOpt = oneshot.SamplingSearch(sampler=init, scrambled=init == "Hammersley")
        # configuration
        self.utility_kind = utility_kind
        self.utility_kappa = utility_kappa
        self.utility_xi = utility_xi
        self.gp_parameters = {} if gp_parameters is None else gp_parameters
        if isinstance(parametrization, p.Parameter) and self.gp_parameters.get("alpha", 0) == 0:
            analysis = p.helpers.analyze(parametrization)
            noisy = not analysis.deterministic
            cont = analysis.continuous
            if noisy or not cont:
                warnings.warn(
                    "Dis-continuous and noisy parametrization require gp_parameters['alpha'] > 0 "
                    "(for your parametrization, continuity={cont} and noisy={noisy}).\n"
                    "Find more information on BayesianOptimization's github.\n"
                    "You should then create a new instance of optimizerlib.ParametrizedBO with appropriate parametrization.",
                    errors.InefficientSettingsWarning,
                )

    @property
    def bo(self) -> BayesianOptimization:
        if self._bo is None:
            bounds = {self._fake_function.key(i): (0.0, 1.0) for i in range(self.dimension)}
            self._bo = BayesianOptimization(self._fake_function, bounds, random_state=self._rng)
            init_budget = max(
                2, int(np.sqrt(self.budget) if self._init_budget is None else self._init_budget)
            )
            if self.gp_parameters is not None:
                self._bo.set_gp_params(**self.gp_parameters)
            # init
            if self._middle_point:
                self._bo.probe([0.5] * self.dimension, lazy=True)
                init_budget -= 1
            if self._InitOpt is not None and init_budget > 0:
                param = p.Array(shape=(self.dimension,)).set_bounds(lower=0, upper=1.0)
                param.random_state = self._rng
                opt = self._InitOpt(param, budget=init_budget)
                for _ in range(init_budget):
                    self._bo.probe(opt.ask().value, lazy=True)
            else:  # default
                for _ in range(init_budget):
                    self._bo.probe(self._bo._space.random_sample(), lazy=True)
        return self._bo

    def _internal_ask_candidate(self) -> p.Parameter:
        util = UtilityFunction(kind=self.utility_kind, kappa=self.utility_kappa, xi=self.utility_xi)
        if self.bo._queue:
            x_probe = next(self.bo._queue)
        else:
            x_probe = self.bo.suggest(util)  # this is time consuming
            x_probe = [x_probe[self._fake_function.key(i)] for i in range(len(x_probe))]
        data = self._transform.backward(np.array(x_probe, copy=False))
        candidate = self.parametrization.spawn_child().set_standardized_data(data)
        candidate._meta["x_probe"] = x_probe
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        if "x_probe" in candidate._meta:
            y = candidate._meta["x_probe"]
        else:
            data = candidate.get_standardized_data(reference=self.parametrization)
            y = self._transform.forward(data)  # tell not asked
        self._fake_function.register(y, -loss)  # minimizing
        self.bo.probe(y, lazy=False)
        # for some unknown reasons, BO wants to evaluate twice the same point,
        # but since it keeps a cache of the values, the registered value is not used
        # so we should clean the "fake" function
        self._fake_function._registered.clear()

    def _internal_provide_recommendation(self) -> tp.Optional[tp.ArrayLike]:
        if not self.archive:
            return None
        return self._transform.backward(
            np.array([self.bo.max["params"][self._fake_function.key(i)] for i in range(self.dimension)])
        )


class ParametrizedBO(base.ConfiguredOptimizer):
    """Bayesian optimization.
    Hyperparameter tuning method, based on statistical modeling of the objective function.
    This class is a wrapper over the `bayes_opt <https://github.com/fmfn/BayesianOptimization>`_ package.

    Parameters
    ----------
    initialization: str
        Initialization algorithms (None, "Hammersley", "random" or "LHS")
    init_budget: int or None
        Number of initialization algorithm steps
    middle_point: bool
        whether to sample the 0 point first
    utility_kind: str
        Type of utility function to use among "ucb", "ei" and "poi"
    utility_kappa: float
        Kappa parameter for the utility function
    utility_xi: float
        Xi parameter for the utility function
    gp_parameters: dict
        dictionnary of parameters for the gaussian process
    """

    no_parallelization = True

    # pylint: disable=unused-argument
    def __init__(
        self,
        *,
        initialization: tp.Optional[str] = None,
        init_budget: tp.Optional[int] = None,
        middle_point: bool = False,
        utility_kind: str = "ucb",  # bayes_opt default
        utility_kappa: float = 2.576,
        utility_xi: float = 0.0,
        gp_parameters: tp.Optional[tp.Dict[str, tp.Any]] = None,
    ) -> None:
        super().__init__(_BO, locals())


BO = ParametrizedBO().set_name("BO", register=True)


class _Chain(base.Optimizer):
    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        *,
        optimizers: tp.Sequence[tp.Union[base.ConfiguredOptimizer, tp.Type[base.Optimizer]]] = [
            LHSSearch,
            DE,
        ],
        budgets: tp.Sequence[tp.Union[str, int]] = (10,),
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        # delayed initialization
        # Either we have the budget for each algorithm, or the last algorithm uses the rest of the budget, so:
        self.optimizers: tp.List[base.Optimizer] = []
        converter = {
            "num_workers": self.num_workers,
            "dimension": self.dimension,
            "half": self.budget // 2 if self.budget else self.num_workers,
            "third": self.budget // 3 if self.budget else self.num_workers,
            "sqrt": int(np.sqrt(self.budget)) if self.budget else self.num_workers,
        }
        self.budgets = [max(1, converter[b]) if isinstance(b, str) else b for b in budgets]
        last_budget = None if self.budget is None else max(4, self.budget - sum(self.budgets))
        assert len(optimizers) == len(self.budgets) + 1
        assert all(
            x in ("third", "half", "dimension", "num_workers", "sqrt") or x > 0 for x in self.budgets
        ), str(self.budgets)
        for opt, optbudget in zip(optimizers, self.budgets + [last_budget]):  # type: ignore
            self.optimizers.append(opt(self.parametrization, budget=optbudget, num_workers=self.num_workers))
        if self.name.startswith("chain"):
            warnings.warn(
                "Chain optimizers are renamed with a capital C for consistency. "
                "Eg: chainCMAPowell becomes ChainCMAPowell",
                errors.NevergradDeprecationWarning,
            )

    def _internal_ask_candidate(self) -> p.Parameter:
        # Which algorithm are we playing with ?
        sum_budget = 0.0
        opt = self.optimizers[0]
        for opt in self.optimizers:
            sum_budget += float("inf") if opt.budget is None else opt.budget
            if self.num_ask < sum_budget:
                break
        # if we are over budget, then use the last one...
        return opt.ask()

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        # Let us inform all concerned algorithms
        sum_budget = 0.0
        for opt in self.optimizers:
            sum_budget += float("inf") if opt.budget is None else opt.budget
            if self.num_tell < sum_budget:
                opt.tell(candidate, loss)


class Chaining(base.ConfiguredOptimizer):
    """
    A chaining consists in running algorithm 1 during T1, then algorithm 2 during T2, then algorithm 3 during T3, etc.
    Each algorithm is fed with what happened before it.

    Parameters
    ----------
    optimizers: list of Optimizer classes
        the sequence of optimizers to use
    budgets: list of int
        the corresponding budgets for each optimizer but the last one

    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        optimizers: tp.Sequence[tp.Union[base.ConfiguredOptimizer, tp.Type[base.Optimizer]]],
        budgets: tp.Sequence[tp.Union[str, int]],
    ) -> None:
        super().__init__(_Chain, locals())


# depreated: old names (need a capital letter for consistency
chainCMAPowell = Chaining([CMA, Powell], ["half"]).set_name("chainCMAPowell", register=True)
chainCMAPowell.no_parallelization = True
chainMetaModelSQP = Chaining([MetaModel, SQP], ["half"]).set_name("chainMetaModelSQP", register=True)
chainMetaModelSQP.no_parallelization = True
chainMetaModelPowell = Chaining([MetaModel, Powell], ["half"]).set_name("chainMetaModelPowell", register=True)
chainMetaModelPowell.no_parallelization = True
chainDiagonalCMAPowell = Chaining([DiagonalCMA, Powell], ["half"]).set_name(
    "chainDiagonalCMAPowell", register=True
)
chainDiagonalCMAPowell.no_parallelization = True
chainNaiveTBPSAPowell = Chaining([NaiveTBPSA, Powell], ["half"]).set_name(
    "chainNaiveTBPSAPowell", register=True
)
chainNaiveTBPSAPowell.no_parallelization = True
chainNaiveTBPSACMAPowell = Chaining([NaiveTBPSA, CMA, Powell], ["third", "third"]).set_name(
    "chainNaiveTBPSACMAPowell", register=True
)
chainNaiveTBPSACMAPowell.no_parallelization = True

# new names
ChainCMAPowell = Chaining([CMA, Powell], ["half"]).set_name("ChainCMAPowell", register=True)
ChainCMAPowell.no_parallelization = True  # TODO make this automatic
ChainMetaModelSQP = Chaining([MetaModel, SQP], ["half"]).set_name("ChainMetaModelSQP", register=True)
ChainMetaModelSQP.no_parallelization = True
ChainMetaModelPowell = Chaining([MetaModel, Powell], ["half"]).set_name("ChainMetaModelPowell", register=True)
ChainMetaModelPowell.no_parallelization = True
ChainDiagonalCMAPowell = Chaining([DiagonalCMA, Powell], ["half"]).set_name(
    "ChainDiagonalCMAPowell", register=True
)
ChainDiagonalCMAPowell.no_parallelization = True
ChainNaiveTBPSAPowell = Chaining([NaiveTBPSA, Powell], ["half"]).set_name(
    "ChainNaiveTBPSAPowell", register=True
)
ChainNaiveTBPSAPowell.no_parallelization = True
ChainNaiveTBPSACMAPowell = Chaining([NaiveTBPSA, CMA, Powell], ["third", "third"]).set_name(
    "ChainNaiveTBPSACMAPowell", register=True
)
ChainNaiveTBPSACMAPowell.no_parallelization = True


@registry.register
class cGA(base.Optimizer):
    """`Compact Genetic Algorithm <https://ieeexplore.ieee.org/document/797971>`_.
    A discrete optimization algorithm, introduced in and often used as a first baseline.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        arity: tp.Optional[int] = None,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        if arity is None:
            all_params = p.helpers.flatten(self.parametrization)
            arity = max(
                len(param.choices) if isinstance(param, p.TransitionChoice) else 500
                for _, param in all_params
            )
        self._arity = arity
        self._penalize_cheap_violations = False  # Not sure this is the optimal decision.
        # self.p[i][j] is the probability that the ith variable has value 0<=j< arity.
        self.p: np.ndarray = np.ones((self.dimension, arity)) / arity
        # Probability increments are of order 1./self.llambda
        # and lower bounded by something of order 1./self.llambda.
        self.llambda = max(num_workers, 40)  # FIXME: no good heuristic ?
        # CGA generates a candidate, then a second candidate;
        # then updates depending on the comparison with the first one. We therefore have to store the previous candidate.
        self._previous_value_candidate: tp.Optional[tp.Tuple[float, np.ndarray]] = None

    def _internal_ask_candidate(self) -> p.Parameter:
        # Multinomial.
        values: tp.List[int] = [
            sum(self._rng.uniform() > cum_proba) for cum_proba in np.cumsum(self.p, axis=1)
        ]
        data = discretization.noisy_inverse_threshold_discretization(values, arity=self._arity, gen=self._rng)
        return self.parametrization.spawn_child().set_standardized_data(data)

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        data = candidate.get_standardized_data(reference=self.parametrization)
        if self._previous_value_candidate is None:
            self._previous_value_candidate = (loss, data)
        else:
            winner, loser = self._previous_value_candidate[1], data
            if self._previous_value_candidate[0] > loss:
                winner, loser = loser, winner
            winner_data = discretization.threshold_discretization(np.asarray(winner.data), arity=self._arity)
            loser_data = discretization.threshold_discretization(np.asarray(loser.data), arity=self._arity)
            for i, _ in enumerate(winner_data):
                if winner_data[i] != loser_data[i]:
                    self.p[i][winner_data[i]] += 1.0 / self.llambda
                    self.p[i][loser_data[i]] -= 1.0 / self.llambda
                    for j in range(len(self.p[i])):
                        self.p[i][j] = max(self.p[i][j], 1.0 / self.llambda)
                    self.p[i] /= sum(self.p[i])
            self._previous_value_candidate = None


class _EMNA(base.Optimizer):
    """Simple Estimation of Multivariate Normal Algorithm (EMNA)."""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        isotropic: bool = True,
        naive: bool = True,
        population_size_adaptation: bool = False,
        initial_popsize: tp.Optional[int] = None,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.isotropic: bool = isotropic
        self.naive: bool = naive
        self.population_size_adaptation = population_size_adaptation
        self.min_coef_parallel_context: int = 8
        # Sigma initialization
        self.sigma: tp.Union[float, np.ndarray]
        if initial_popsize is None:
            initial_popsize = self.dimension
        if self.isotropic:
            self.sigma = 1.0
        else:
            self.sigma = np.ones(self.dimension)
        # population size and parent size initializations
        self.popsize = _PopulationSizeController(
            llambda=4 * initial_popsize, mu=initial_popsize, dimension=self.dimension, num_workers=num_workers
        )
        if not self.population_size_adaptation:
            self.popsize.mu = max(16, self.dimension)
            self.popsize.llambda = 4 * self.popsize.mu
            self.popsize.llambda = max(self.popsize.llambda, num_workers)
            if budget is not None and self.popsize.llambda > budget:
                self.popsize.llambda = budget
                self.popsize.mu = self.popsize.llambda // 4
                warnings.warn(
                    "Budget may be too small in front of the dimension for EMNA",
                    errors.InefficientSettingsWarning,
                )
        self.current_center: np.ndarray = np.zeros(self.dimension)
        # population
        self.parents: tp.List[p.Parameter] = [self.parametrization]
        self.children: tp.List[p.Parameter] = []

    def recommend(self) -> p.Parameter:
        if self.naive:
            return self.current_bests["optimistic"].parameter
        else:
            # This is NOT the naive version. We deal with noise.
            out = self.parametrization.spawn_child()
            with p.helpers.deterministic_sampling(out):
                out.set_standardized_data(self.current_center)
            return out

    def _internal_ask_candidate(self) -> p.Parameter:
        sigma_tmp = self.sigma
        if (
            self.population_size_adaptation
            and self.popsize.llambda < self.min_coef_parallel_context * self.dimension
        ):
            sigma_tmp = self.sigma * np.exp(self._rng.normal(0, 1) / np.sqrt(self.dimension))
        individual = self.current_center + sigma_tmp * self._rng.normal(0, 1, self.dimension)
        parent = self.parents[self.num_ask % len(self.parents)]
        candidate = parent.spawn_child().set_standardized_data(individual, reference=self.parametrization)
        if parent is self.parametrization:
            candidate.heritage["lineage"] = candidate.uid
        candidate._meta["sigma"] = sigma_tmp
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        if self.population_size_adaptation:
            self.popsize.add_value(loss)
        self.children.append(candidate)
        if len(self.children) >= self.popsize.llambda:
            # Sorting the population.
            self.children.sort(key=base._loss)
            # Computing the new parent.
            self.parents = self.children[: self.popsize.mu]
            self.children = []
            self.current_center = (
                sum(  # type: ignore
                    c.get_standardized_data(reference=self.parametrization) for c in self.parents
                )
                / self.popsize.mu
            )
            if self.population_size_adaptation:
                if (
                    self.popsize.llambda < self.min_coef_parallel_context * self.dimension
                ):  # Population size not large enough for emna
                    self.sigma = np.exp(
                        np.sum(
                            np.log([c._meta["sigma"] for c in self.parents]),
                            axis=0 if self.isotropic else None,
                        )
                        / self.popsize.mu
                    )
                else:
                    stdd = [
                        (
                            self.parents[i].get_standardized_data(reference=self.parametrization)
                            - self.current_center
                        )
                        ** 2
                        for i in range(self.popsize.mu)
                    ]
                    self.sigma = np.sqrt(
                        np.sum(stdd) / (self.popsize.mu * (self.dimension if self.isotropic else 1))
                    )
            else:
                # EMNA update
                stdd = [
                    (
                        self.parents[i].get_standardized_data(reference=self.parametrization)
                        - self.current_center
                    )
                    ** 2
                    for i in range(self.popsize.mu)
                ]
                self.sigma = np.sqrt(
                    np.sum(stdd, axis=0 if self.isotropic else None)
                    / (self.popsize.mu * (self.dimension if self.isotropic else 1))
                )

            if self.num_workers / self.dimension > 32:  # faster decrease of sigma if large parallel context
                imp = max(1, (np.log(self.popsize.llambda) / 2) ** (1 / self.dimension))
                self.sigma /= imp

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        raise errors.TellNotAskedNotSupportedError


class EMNA(base.ConfiguredOptimizer):
    """Estimation of Multivariate Normal Algorithm
    This algorithm is quite efficient in a parallel context, i.e. when
    the population size is large.

    Parameters
    ----------
    isotropic: bool
        isotropic version on EMNA if True, i.e. we have an
        identity matrix for the Gaussian, else  we here consider the separable
        version, meaning we have a diagonal matrix for the Gaussian (anisotropic)
    naive: bool
        set to False for noisy problem, so that the best points will be an
        average of the final population.
    population_size_adaptation: bool
        population size automatically adapts to the landscape
    initial_popsize: Optional[int]
        initial (and minimal) population size (default: 4 x dimension)
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        *,
        isotropic: bool = True,
        naive: bool = True,
        population_size_adaptation: bool = False,
        initial_popsize: tp.Optional[int] = None,
    ) -> None:
        super().__init__(_EMNA, locals())


NaiveIsoEMNA = EMNA().set_name("NaiveIsoEMNA", register=True)


# Discussions with Jialin Liu and Fabien Teytaud helped the following development.
# This includes discussion at Dagstuhl's 2019 seminars on randomized search heuristics and computational intelligence in games.
@registry.register
class NGOptBase(base.Optimizer):
    """Nevergrad optimizer by competence map."""

    # pylint: disable=too-many-branches
    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        analysis = p.helpers.analyze(self.parametrization)
        funcinfo = self.parametrization.function
        self.has_noise = not (analysis.deterministic and funcinfo.deterministic)
        # The noise coming from discrete variables goes to 0.
        self.noise_from_instrumentation = self.has_noise and funcinfo.deterministic
        self.fully_continuous = analysis.continuous
        all_params = p.helpers.flatten(self.parametrization)
        # figure out if there is any discretization layers
        int_layers = list(
            itertools.chain.from_iterable([_layering.Int.filter_from(x) for _, x in all_params])
        )
        int_layers = [x for x in int_layers if x.arity is not None]  # only "Choice" instances for now
        self.has_discrete_not_softmax = any(
            not isinstance(lay, _datalayers.SoftmaxSampling) for lay in int_layers
        )
        self._has_discrete = bool(int_layers)
        self._arity: int = max((lay.arity for lay in int_layers), default=-1)  # type: ignore
        if self.fully_continuous:
            self._arity = -1
        self._optim: tp.Optional[base.Optimizer] = None
        self._constraints_manager.update(
            max_trials=1000,
            penalty_factor=1.0,
            penalty_exponent=1.01,
        )

    @property
    def optim(self) -> base.Optimizer:
        if self._optim is None:
            self._optim = self._select_optimizer_cls()(self.parametrization, self.budget, self.num_workers)
            optim = self._optim if not isinstance(self._optim, NGOptBase) else self._optim.optim
            logger.debug("%s selected %s optimizer.", *(x.name for x in (self, optim)))
        return self._optim

    def _select_optimizer_cls(self) -> base.OptCls:
        # pylint: disable=too-many-nested-blocks
        assert self.budget is not None
        if self.has_noise and self.has_discrete_not_softmax:
            # noise and discrete: let us merge evolution and bandits.
            cls: base.OptCls = DoubleFastGADiscreteOnePlusOne if self.dimension < 60 else CMA
        else:
            if self.has_noise and self.fully_continuous:
                # This is the real of population control. FIXME: should we pair with a bandit ?
                cls = TBPSA
            else:
                if (
                    self.has_discrete_not_softmax
                    or not self.parametrization.function.metrizable
                    or not self.fully_continuous
                ):
                    cls = DoubleFastGADiscreteOnePlusOne
                else:
                    if self.num_workers > self.budget / 5:
                        if self.num_workers > self.budget / 2.0 or self.budget < self.dimension:
                            cls = MetaRecentering
                        else:
                            cls = NaiveTBPSA
                    else:
                        # Possibly a good idea to go memetic for large budget, but something goes wrong for the moment.
                        if (
                            self.num_workers == 1 and self.budget > 6000 and self.dimension > 7
                        ):  # Let us go memetic.
                            cls = ChainCMAPowell
                        else:
                            if self.num_workers == 1 and self.budget < self.dimension * 30:
                                # One plus one so good in large ratio "dimension / budget".
                                cls = OnePlusOne if self.dimension > 30 else Cobyla
                            else:
                                # DE is great in such a case (?).
                                cls = (
                                    DE if self.dimension > 2000 else CMA if self.dimension > 1 else OnePlusOne
                                )
        return cls

    def _internal_ask_candidate(self) -> p.Parameter:
        return self.optim.ask()

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        self.optim.tell(candidate, loss)

    def recommend(self) -> p.Parameter:
        return self.optim.recommend()

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        self.optim.tell(candidate, loss)


@registry.register
class Shiwa(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing youe own competence map."""

    def _select_optimizer_cls(self) -> base.OptCls:
        optCls: base.OptCls = NGOptBase
        funcinfo = self.parametrization.function
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        elif self.dimension >= 60 and not funcinfo.metrizable:
            optCls = CMA
        return optCls


@registry.register
class NGO(NGOptBase):  # compatibility
    pass


@registry.register
class NGOpt4(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing youe own competence map."""

    def _select_optimizer_cls(self) -> base.OptCls:
        self.fully_continuous = (
            self.fully_continuous and not self.has_discrete_not_softmax and self._arity < 0
        )
        budget, num_workers = self.budget, self.num_workers
        funcinfo = self.parametrization.function
        assert budget is not None
        optimClass: base.OptCls
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            mutation = "portfolio" if budget > 1000 else "discrete"
            optimClass = ParametrizedOnePlusOne(
                crossover=True, mutation=mutation, noise_handling="optimistic"
            )
        elif self._arity > 0:
            if self._arity == 2:
                optimClass = DiscreteOnePlusOne
            else:
                optimClass = AdaptiveDiscreteOnePlusOne if self._arity < 5 else CMandAS2
        else:
            # pylint: disable=too-many-nested-blocks
            if self.has_noise and self.fully_continuous and self.dimension > 100:
                # Waow, this is actually a discrete algorithm.
                optimClass = ConfSplitOptimizer(
                    num_optims=13, progressive=True, multivariate_optimizer=OptimisticDiscreteOnePlusOne
                )
            else:
                if self.has_noise and self.fully_continuous:
                    if budget > 100:
                        optimClass = (
                            OnePlusOne if self.noise_from_instrumentation or self.num_workers > 1 else SQP
                        )
                    else:
                        optimClass = OnePlusOne
                else:
                    if self.has_discrete_not_softmax or not funcinfo.metrizable or not self.fully_continuous:
                        optimClass = DoubleFastGADiscreteOnePlusOne
                    else:
                        if num_workers > budget / 5:
                            if num_workers > budget / 2.0 or budget < self.dimension:
                                optimClass = MetaTuneRecentering
                            elif self.dimension < 5 and budget < 100:
                                optimClass = DiagonalCMA
                            elif self.dimension < 5 and budget < 500:
                                optimClass = Chaining([DiagonalCMA, MetaModel], [100])
                            else:
                                optimClass = NaiveTBPSA
                        else:
                            # Possibly a good idea to go memetic for large budget, but something goes wrong for the moment.
                            if (
                                num_workers == 1 and budget > 6000 and self.dimension > 7
                            ):  # Let us go memetic.
                                optimClass = ChainNaiveTBPSACMAPowell
                            else:
                                if num_workers == 1 and budget < self.dimension * 30:
                                    if (
                                        self.dimension > 30
                                    ):  # One plus one so good in large ratio "dimension / budget".
                                        optimClass = OnePlusOne
                                    elif self.dimension < 5:
                                        optimClass = MetaModel
                                    else:
                                        optimClass = Cobyla
                                else:
                                    if self.dimension > 2000:  # DE is great in such a case (?).
                                        optimClass = DE
                                    else:
                                        if self.dimension < 10 and budget < 500:
                                            optimClass = MetaModel
                                        else:
                                            if (
                                                self.dimension > 40
                                                and num_workers > self.dimension
                                                and budget < 7 * self.dimension ** 2
                                            ):
                                                optimClass = DiagonalCMA
                                            elif (
                                                3 * num_workers > self.dimension ** 2
                                                and budget > self.dimension ** 2
                                            ):
                                                optimClass = MetaModel
                                            else:
                                                optimClass = CMA
        return optimClass


@registry.register
class NGOpt8(NGOpt4):
    """Nevergrad optimizer by competence map. You might modify this one for designing youe own competence map."""

    def _select_optimizer_cls(self) -> base.OptCls:
        # Extracting info as far as possible.
        assert self.budget is not None
        funcinfo = self.parametrization.function
        optimClass: base.OptCls
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            if self.budget > 10000:
                optimClass = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
            else:
                optimClass = ParametrizedOnePlusOne(
                    crossover=True, mutation="discrete", noise_handling="optimistic"
                )
        elif self._arity > 0:
            if self.budget < 1000 and self.num_workers == 1:
                optimClass = DiscreteBSOOnePlusOne
            elif self.num_workers > 2:
                optimClass = CMandAS2  # type: ignore
            else:
                optimClass = super()._select_optimizer_cls()
        else:
            if (
                not (self.has_noise and self.fully_continuous and self.dimension > 100)
                and not (self.has_noise and self.fully_continuous)
                and not (self.num_workers > self.budget / 5)
                and (self.num_workers == 1 and self.budget > 6000 and self.dimension > 7)
            ):
                optimClass = ChainMetaModelPowell
            else:
                optimClass = super()._select_optimizer_cls()

        return optimClass

    def _num_objectives_set_callback(self) -> None:
        super()._num_objectives_set_callback()
        if self.num_objectives > 1:
            if self.noise_from_instrumentation or not self.has_noise:
                # override at runtime
                self._optim = DE(self.parametrization, self.budget, self.num_workers)


@registry.register
class NGOpt10(NGOpt8):
    def _select_optimizer_cls(self) -> base.OptCls:
        if not self.has_noise and self._arity > 0:
            return DiscreteLenglerOnePlusOne
        else:
            return super()._select_optimizer_cls()

    def recommend(self) -> p.Parameter:
        return base.Optimizer.recommend(self)


@registry.register
class NGOpt(NGOpt10):
    pass


class _MSR(CM):
    """This code applies multiple copies of NGOpt with random weights for the different objective functions.

    Variants dedicated to multiobjective optimization by multiple singleobjective optimization.
    """

    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        num_single_runs: int = 9,
        base_optimizer: base.OptCls = NGOpt,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.num_optims = num_single_runs
        self.optims = [
            base_optimizer(
                self.parametrization,
                budget=1 + (budget // self.num_optims) if budget is not None else None,
                num_workers=num_workers,
            )
            for _ in range(self.num_optims)
        ]
        self.coeffs: tp.List[np.ndarray] = []

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        if not self.coeffs:
            self.coeffs = [
                self.parametrization.random_state.uniform(size=self.num_objectives)
                for _ in range(self.num_optims)
            ]
        for coeffs, opt in zip(self.coeffs, self.optims):
            this_loss = np.sum(loss * coeffs)
            opt.tell(candidate, this_loss)


class MultipleSingleRuns(base.ConfiguredOptimizer):
    """Multiple single-objective runs, in particular for multi-objective optimization.
    Parameters
    ----------
    num_single_runs: int
        number of single runs.
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        *,
        num_single_runs: int = 9,
        base_optimizer: base.OptCls = NGOpt,
    ) -> None:
        super().__init__(_MSR, locals())
