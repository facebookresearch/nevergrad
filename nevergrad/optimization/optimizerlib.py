# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import typing as tp  # from now on, favor using tp.Dict etc instead of Dict
from typing import Optional, List, Dict, Tuple, Callable, Any
from collections import deque
import warnings
import cma
import numpy as np
from bayes_opt import UtilityFunction
from bayes_opt import BayesianOptimization
from nevergrad.parametrization import parameter as p
from nevergrad.parametrization import transforms
from nevergrad.parametrization import discretization
from nevergrad.parametrization import helpers as paramhelpers
from nevergrad.common.typetools import ArrayLike
from . import base
from . import mutations
from .base import registry as registry
from .base import addCompare  # pylint: disable=unused-import
from .base import InefficientSettingsWarning as InefficientSettingsWarning
from .base import IntOrParameter
from . import sequences


# families of optimizers
# pylint: disable=unused-wildcard-import,wildcard-import,too-many-lines,too-many-arguments
from .differentialevolution import *  # noqa: F403
from .es import *  # noqa: F403
from .oneshot import *  # noqa: F403
from .recastlib import *  # noqa: F403


# # # # # optimizers # # # # #


class _OnePlusOne(base.Optimizer):
    """Simple but sometimes powerful optimization algorithm.

    We use the one-fifth adaptation rule, going back to Schumer and Steiglitz (1968).
    It was independently rediscovered by Devroye (1972) and Rechenberg (1973).
    We use asynchronous updates, so that the 1+1 can actually be parallel and even
    performs quite well in such a context - this is naturally close to 1+lambda.
    """

    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: Optional[int] = None,
        num_workers: int = 1,
        *,
        noise_handling: tp.Optional[tp.Union[str, tp.Tuple[str, float]]] = None,
        mutation: str = "gaussian",
        crossover: bool = False
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._sigma: float = 1
        # configuration
        if noise_handling is not None:
            if isinstance(noise_handling, str):
                assert noise_handling in ["random", "optimistic"], f"Unkwnown noise handling: '{noise_handling}'"
            else:
                assert isinstance(noise_handling, tuple), "noise_handling must be a string or  a tuple of type (strategy, factor)"
                assert noise_handling[1] > 0.0, "the factor must be a float greater than 0"
                assert noise_handling[0] in ["random", "optimistic"], f"Unkwnown noise handling: '{noise_handling}'"
        assert mutation in ["gaussian", "cauchy", "discrete", "fastga", "doublefastga", "portfolio"], f"Unkwnown mutation: '{mutation}'"
        self.noise_handling = noise_handling
        self.mutation = mutation
        self.crossover = crossover

    def _internal_ask(self) -> ArrayLike:
        # pylint: disable=too-many-return-statements, too-many-branches
        noise_handling = self.noise_handling
        if not self._num_ask:
            return np.zeros(self.dimension)  # type: ignore
        # for noisy version
        if noise_handling is not None:
            limit = (0.05 if isinstance(noise_handling, str) else noise_handling[1]) * len(self.archive) ** 3
            strategy = noise_handling if isinstance(noise_handling, str) else noise_handling[0]
            if self._num_ask <= limit:
                if strategy in ["cubic", "random"]:
                    idx = self._rng.choice(len(self.archive))
                    return np.frombuffer(list(self.archive.bytesdict.keys())[idx])  # type: ignore
                elif strategy == "optimistic":
                    return self.current_bests["optimistic"].x
        # crossover
        mutator = mutations.Mutator(self._rng)
        if self.crossover and self._num_ask % 2 == 1 and len(self.archive) > 2:
            return mutator.crossover(self.current_bests["pessimistic"].x, mutator.get_roulette(self.archive, num=2))
        # mutating
        mutation = self.mutation
        pessimistic = self.current_bests["pessimistic"].x
        if mutation == "gaussian":  # standard case
            return pessimistic + self._sigma * self._rng.normal(0, 1, self.dimension)  # type: ignore
        elif mutation == "cauchy":
            return pessimistic + self._sigma * self._rng.standard_cauchy(self.dimension)  # type: ignore
        elif mutation == "crossover":
            if self._num_ask % 2 == 0 or len(self.archive) < 3:
                return mutator.portfolio_discrete_mutation(pessimistic)
            else:
                return mutator.crossover(pessimistic, mutator.get_roulette(self.archive, num=2))
        else:
            func: Callable[[ArrayLike], ArrayLike] = {  # type: ignore
                "discrete": mutator.discrete_mutation,
                "fastga": mutator.doerr_discrete_mutation,
                "doublefastga": mutator.doubledoerr_discrete_mutation,
                "portfolio": mutator.portfolio_discrete_mutation,
            }[mutation]
            return func(self.current_bests["pessimistic"].x)

    def _internal_tell(self, x: ArrayLike, value: float) -> None:
        # only used for cauchy and gaussian
        self._sigma *= 2.0 if value <= self.current_bests["pessimistic"].mean else 0.84


class ParametrizedOnePlusOne(base.ConfiguredOptimizer):
    """Simple but sometimes powerfull class of optimization algorithm.
    We use asynchronous updates, so that the 1+1 can actually be parallel and even
    performs quite well in such a context - this is naturally close to 1+lambda.

    Parameters
    ----------
    noise_handling: str or Tuple[str, float]
        Method for handling the noise. The name can be:

        - `"random"`: a random point is reevaluated regularly
        - `"optimistic"`: the best optimistic point is reevaluated regularly, optimism in front of uncertainty
        - a coefficient can to tune the regularity of these reevaluations (default .05)
    mutation: str
        One of the available mutations from:

        - `"gaussian"`: standard mutation by adding a Gaussian random variable (with progressive
          widening) to the best pessimistic point
        - `"cauchy"`: same as Gaussian but with a Cauchy distribution.
        - `"discrete"`: TODO
        - `"fastga"`: FastGA mutations from the current best
        - `"doublefastga"`: double-FastGA mutations from the current best (Doerr et al, Fast Genetic Algorithms, 2017)
        - `"portfolio"`: Random number of mutated bits (called niform mixing in
          Dang & Lehre "Self-adaptation of Mutation Rates in Non-elitist Population", 2016)

    crossover: bool
        whether to add a genetic crossover step every other iteration.

    Notes
    -----
    For the noisy case, we use the one-fifth adaptation rule,
    going back to Schumer and Steiglitz (1968).
    It was independently rediscovered by Devroye (1972) and Rechenberg (1973).
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        *,
        noise_handling: tp.Optional[tp.Union[str, tp.Tuple[str, float]]] = None,
        mutation: str = "gaussian",
        crossover: bool = False
    ) -> None:
        super().__init__(_OnePlusOne, locals())


OnePlusOne = ParametrizedOnePlusOne().set_name("OnePlusOne", register=True)
NoisyOnePlusOne = ParametrizedOnePlusOne(noise_handling="random").set_name("NoisyOnePlusOne", register=True)
DiscreteOnePlusOne = ParametrizedOnePlusOne(mutation="discrete").set_name("DiscreteOnePlusOne", register=True)
CauchyOnePlusOne = ParametrizedOnePlusOne(mutation="cauchy").set_name("CauchyOnePlusOne", register=True)
OptimisticNoisyOnePlusOne = ParametrizedOnePlusOne(
    noise_handling="optimistic").set_name("OptimisticNoisyOnePlusOne", register=True)
OptimisticDiscreteOnePlusOne = ParametrizedOnePlusOne(noise_handling="optimistic", mutation="discrete").set_name(
    "OptimisticDiscreteOnePlusOne", register=True
)
NoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(noise_handling=("random", 1.0), mutation="discrete").set_name(
    "NoisyDiscreteOnePlusOne", register=True
)
DoubleFastGADiscreteOnePlusOne = ParametrizedOnePlusOne(
    mutation="doublefastga").set_name("DoubleFastGADiscreteOnePlusOne", register=True)
RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(
    crossover=True, mutation="portfolio", noise_handling="optimistic"
).set_name("RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne", register=True)


# pylint: too-many-arguments, too-many-instance-attributes
class _CMA(base.Optimizer):

    def __init__(
            self,
            parametrization: IntOrParameter,
            budget: Optional[int] = None,
            num_workers: int = 1,
            scale: float = 1.0,
            popsize: Optional[int] = None,
            diagonal: bool = False,
            fcmaes: bool = False
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._scale = scale
        self._popsize = popsize
        self._diagonal = diagonal
        self._fcmaes = fcmaes
        self._es: Optional[cma.CMAEvolutionStrategy] = None
        # delay initialization to ease implementation of variants
        self.listx: tp.List[ArrayLike] = []
        self.listy: tp.List[float] = []
        self.to_be_asked: tp.Deque[np.ndarray] = deque()

    @property
    def es(self) -> tp.Any:  # typing not possible since cmaes not imported :(
        if self._es is None:
            popsize = max(self.num_workers, 4 + int(3 * np.log(self.dimension))) if self._popsize is None else self._popsize
            if self._fcmaes:
                try:
                    from fcmaes import cmaes
                except ImportError as e:
                    raise ImportError("Please install fcmaes (pip install fcmaes) to use FCMA optimizers") from e
                self._es = cmaes.Cmaes(x0=np.zeros(self.dimension, dtype=np.float),
                                       input_sigma=self._scale,
                                       popsize=popsize, randn=self._rng.randn)
            else:
                inopts = {"popsize": popsize, "randn": self._rng.randn, "CMA_diagonal": self._diagonal, "verbose": 0}
                self._es = cma.CMAEvolutionStrategy(x0=np.zeros(self.dimension, dtype=np.float), sigma0=self._scale, inopts=inopts)
        return self._es

    def _internal_ask(self) -> ArrayLike:
        if not self.to_be_asked:
            self.to_be_asked.extend(self.es.ask())
        return self.to_be_asked.popleft()

    def _internal_tell(self, x: ArrayLike, value: float) -> None:
        self.listx += [x]
        self.listy += [value]
        if len(self.listx) >= self.es.popsize:
            try:
                if self._fcmaes:
                    self.es.tell(self.listy, self.listx)
                else:
                    self.es.tell(self.listx, self.listy)
            except RuntimeError:
                pass
            else:
                self.listx = []
                self.listy = []

    def _internal_provide_recommendation(self) -> ArrayLike:
        if self._es is None:
            return self.current_bests["pessimistic"].x
        cma_best: tp.Optional[ArrayLike] = self.es.best_x if self._fcmaes else self.es.result.xbest
        if cma_best is None:
            return self.current_bests["pessimistic"].x
        return cma_best


class ParametrizedCMA(base.ConfiguredOptimizer):
    """CMA-ES optimizer, wrapping external implementation: https://github.com/CMA-ES/pycma

    Parameters
    ----------
    scale: float
        scale of the search
    popsize: Optional[int] = None
        population size, should be n * self.num_workers for int n >= 1.
        default is max(self.num_workers, 4 + int(3 * np.log(self.dimension)))

    diagonal: bool
        use the diagonal version of CMA (advised in big dimension)
    fcmaes: bool = False
        use fast implementation, doesn't support diagonal=True.
        produces equivalent results, preferable for high dimensions or
        if objective function evaluation is fast.
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        *,
        scale: float = 1.0,
        popsize: Optional[int] = None,
        diagonal: bool = False,
        fcmaes: bool = False
    ) -> None:
        super().__init__(_CMA, locals())
        if fcmaes:
            if diagonal:
                raise RuntimeError("fcmaes doesn't support diagonal=True, use fcmaes=False")


CMA = ParametrizedCMA().set_name("CMA", register=True)
DiagonalCMA = ParametrizedCMA(diagonal=True).set_name("DiagonalCMA", register=True)
FCMA = ParametrizedCMA(fcmaes=True).set_name("FCMA", register=True)


class _PopulationSizeController:
    """Population control scheme for TBPSA and EDA
    """

    def __init__(self, llambda: int, mu: int, dimension: int, num_workers: int = 1) -> None:
        self.llambda = max(llambda, num_workers)
        self.mu = mu
        self.dimension = dimension
        self.num_workers = num_workers
        self._loss_record: tp.List[float] = []

    def add_value(self, value: float) -> None:
        self._loss_record += [value]
        if len(self._loss_record) >= 5 * self.llambda:
            first_fifth = self._loss_record[: self.llambda]
            last_fifth = self._loss_record[-self.llambda:]
            means = [sum(fitnesses) / float(self.llambda) for fitnesses in [first_fifth, last_fifth]]
            stds = [np.std(fitnesses) / np.sqrt(self.llambda - 1) for fitnesses in [first_fifth, last_fifth]]
            z = (means[0] - means[1]) / (np.sqrt(stds[0] ** 2 + stds[1] ** 2))
            if z < 2.0:
                self.mu *= 2
            else:
                self.mu = int(self.mu * 0.84)
                if self.mu < self.dimension:
                    self.mu = self.dimension
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
    """

    _POPSIZE_ADAPTATION = False
    _COVARIANCE_MEMORY = False

    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.sigma = 1
        self.covariance = np.identity(self.dimension)
        dim = self.dimension
        self.popsize = _PopulationSizeController(llambda=4 * dim, mu=dim, dimension=dim, num_workers=num_workers)
        self.current_center: np.ndarray = np.zeros(self.dimension)
        # Population
        self.children: tp.List[p.Parameter] = []
        self.parents: List[p.Parameter] = [self.parametrization]  # for transfering heritage (checkpoints in PBT)

    def _internal_provide_recommendation(self) -> ArrayLike:  # This is NOT the naive version. We deal with noise.
        return self.current_center

    def _internal_ask_candidate(self) -> p.Parameter:
        mutated_sigma = self.sigma * np.exp(self._rng.normal(0, 1) / np.sqrt(self.dimension))
        assert len(self.current_center) == len(self.covariance), [self.dimension, self.current_center, self.covariance]
        data = mutated_sigma * self._rng.multivariate_normal(self.current_center, self.covariance)
        parent = self.parents[self.num_ask % len(self.parents)]
        candidate = parent.spawn_child().set_standardized_data(data, reference=self.parametrization)
        if parent is self.parametrization:
            candidate.heritage["lineage"] = candidate.uid  # for tracking
        candidate._meta["sigma"] = mutated_sigma
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, value: float) -> None:
        candidate._meta["loss"] = value
        self.children.append(candidate)
        if self._POPSIZE_ADAPTATION:
            self.popsize.add_value(value)
        if len(self.children) >= self.popsize.llambda:
            self.children = sorted(self.children, key=lambda c: c._meta["loss"])
            population_data = [c.get_standardized_data(reference=self.parametrization) for c in self.children]
            self.covariance *= 0.9 if self._COVARIANCE_MEMORY else 0
            self.covariance += 0.1 * np.cov(np.array(population_data).T)
            # Computing the new parent
            mu = self.popsize.mu
            arrays = [d for d in population_data[:mu]]
            self.current_center = sum(arrays) / mu  # type: ignore
            self.sigma = np.exp(sum([np.log(c._meta["sigma"]) for c in self.children[:mu]]) / mu)
            self.parents = self.children[:mu]
            self.children = []

    def _internal_tell_not_asked(self, candidate: p.Parameter, value: float) -> None:
        raise base.TellNotAskedNotSupportedError


@registry.register
class PCEDA(EDA):
    _POPSIZE_ADAPTATION = True
    _COVARIANCE_MEMORY = False


@registry.register
class MPCEDA(EDA):
    _POPSIZE_ADAPTATION = True
    _COVARIANCE_MEMORY = True


@registry.register
class MEDA(EDA):
    _POPSIZE_ADAPTATION = False
    _COVARIANCE_MEMORY = True


class _TBPSA(base.Optimizer):
    """Test-based population-size adaptation.

    Population-size equal to lambda = 4 x dimension.
    Test by comparing the first fifth and the last fifth of the 5lambda evaluations.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self,
                 parametrization: IntOrParameter,
                 budget: Optional[int] = None,
                 num_workers: int = 1,
                 naive: bool = True
                 ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.sigma = 1
        self.naive = naive
        dim = self.dimension
        self.popsize = _PopulationSizeController(llambda=4 * dim, mu=dim, dimension=dim, num_workers=num_workers)
        self.current_center: np.ndarray = np.zeros(self.dimension)
        # population
        self.parents: List[p.Parameter] = [self.parametrization]  # for transfering heritage (checkpoints in PBT)
        self.children: List[p.Parameter] = []

    def _internal_provide_recommendation(self) -> ArrayLike:  # This is NOT the naive version. We deal with noise.
        if self.naive:
            return self.current_bests["optimistic"].x
        else:
            return self.current_center

    def _internal_ask_candidate(self) -> p.Parameter:
        mutated_sigma = self.sigma * np.exp(self._rng.normal(0, 1) / np.sqrt(self.dimension))
        individual = self.current_center + mutated_sigma * self._rng.normal(0, 1, self.dimension)
        parent = self.parents[self.num_ask % len(self.parents)]
        candidate = parent.spawn_child().set_standardized_data(individual, reference=self.parametrization)
        if parent is self.parametrization:
            candidate.heritage["lineage"] = candidate.uid  # for tracking
        candidate._meta["sigma"] = mutated_sigma
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, value: float) -> None:
        candidate._meta["loss"] = value
        self.popsize.add_value(value)
        self.children.append(candidate)
        if len(self.children) >= self.popsize.llambda:
            # Sorting the population.
            self.children.sort(key=lambda c: c._meta["loss"])
            # Computing the new parent.
            self.parents = self.children[: self.popsize.mu]
            self.children = []
            self.current_center = sum(c.get_standardized_data(reference=self.parametrization)  # type: ignore
                                      for c in self.parents) / self.popsize.mu
            self.sigma = np.exp(np.sum(np.log([c._meta["sigma"] for c in self.parents])) / self.popsize.mu)

    def _internal_tell_not_asked(self, candidate: p.Parameter, value: float) -> None:
        data = candidate.get_standardized_data(reference=self.parametrization)
        sigma = np.linalg.norm(data - self.current_center) / np.sqrt(self.dimension)  # educated guess
        candidate._meta["sigma"] = sigma
        self._internal_tell_candidate(candidate, value)  # go through standard pipeline


class ParametrizedTBPSA(base.ConfiguredOptimizer):
    """Test-based population-size adaptation.
    This algorithm is robust, and perfoms well for noisy problems and in large dimension

    Parameters
    ----------
    naive: bool
        set to False for noisy problem, so that the best points will be an
        average of the final population.
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        *,
        naive: bool = True
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

    def _internal_ask(self) -> ArrayLike:
        if 20 * self._num_ask >= len(self.archive) ** 3:
            return self._rng.normal(0, 1, self.dimension)  # type: ignore
        if self._rng.choice([True, False]):
            # numpy does not accept choice on list of tuples, must choose index instead
            idx = self._rng.choice(len(self.archive))
            return np.frombuffer(list(self.archive.bytesdict.keys())[idx])  # type: ignore
        return self.current_bests["optimistic"].x


@registry.register
class PSO(base.Optimizer):
    """Partially following SPSO2011. However, no randomization of the population order.

    Note
    ----
    M. Zambrano-Bigiarini, M. Clerc and R. Rojas,
    Standard Particle Swarm Optimisation 2011 at CEC-2013: A baseline for future PSO improvements,
    2013 IEEE Congress on Evolutionary Computation, Cancun, 2013, pp. 2337-2344.
    https://ieeexplore.ieee.org/document/6557848
    """
    # TODO: the initial speed is probably way too big
    # the recommendation test requires 200 iterations for the mutation to actually be useful

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: Optional[int] = None,
        num_workers: int = 1,
        transform: str = "arctan",
        wide: bool = False,  # legacy, to be removed if not needed anymore
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        if budget is not None and budget < 60:
            warnings.warn("PSO is inefficient with budget < 60", base.InefficientSettingsWarning)
        cases: tp.Dict[str, tp.Tuple[tp.Optional[float], transforms.Transform]] = dict(
            arctan=(0, transforms.ArctanBound(0, 1)),
            identity=(None, transforms.Affine(1, 0)),
            gaussian=(1e-10, transforms.CumulativeDensity()),
        )
        # eps is used for clipping to make sure it is admissible
        self._eps, self._transform = cases[transform]
        self._wide = wide
        self.llambda = max(40, num_workers)
        self._uid_queue = base.utils.UidQueue()
        self.population: tp.Dict[str, p.Parameter] = {}
        self._best = self.parametrization.spawn_child()
        self._omega = 0.5 / np.log(2.0)
        self._phip = 0.5 + np.log(2.0)
        self._phig = 0.5 + np.log(2.0)

    def _internal_ask_candidate(self) -> p.Parameter:
        # population is increased only if queue is empty (otherwise tell_not_asked does not work well at the beginning)
        if len(self.population) < self.llambda:
            param = self.parametrization
            if self._wide:
                # old initialization below seeds in the while R space, while other algorithms use normal distrib
                data = self._transform.backward(self._rng.uniform(0, 1, self.dimension))
                candidate = param.spawn_child().set_standardized_data(data, reference=param)
                candidate.heritage["lineage"] = candidate.uid
            else:
                candidate = param.sample()
            self.population[candidate.uid] = candidate
            dim = self.parametrization.dimension
            candidate.heritage["speed"] = self._rng.normal(size=dim) if self._eps is None else self._rng.uniform(-1, 1, dim)
            self._uid_queue.asked.add(candidate.uid)
            return candidate
        uid = self._uid_queue.ask()
        candidate = self._spawn_mutated_particle(self.population[uid])
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
        speed = self._omega * speed + self._phip * rp * (parent_best_x - x) + self._phig * rg * (global_best_x - x)
        data = speed + x
        if self._eps is not None:
            data = np.clip(data, self._eps, 1 - self._eps)
        data = self._transform.backward(data)
        new_part = particle.spawn_child().set_standardized_data(data, reference=self.parametrization)
        new_part.heritage["speed"] = speed
        return new_part

    def _internal_provide_recommendation(self) -> ArrayLike:
        return self._best.get_standardized_data(reference=self.parametrization)

    def _internal_tell_candidate(self, candidate: p.Parameter, value: float) -> None:
        uid = candidate.heritage["lineage"]
        if uid not in self.population:
            self._internal_tell_not_asked(candidate, value)
            return
        candidate._meta["loss"] = value
        self._uid_queue.tell(uid)
        self.population[uid] = candidate
        if value < self._best._meta.get("loss", float("inf")):
            self._best = candidate
        if value <= candidate.heritage.get("best_parent", candidate)._meta["loss"]:
            candidate.heritage["best_parent"] = candidate

    def _internal_tell_not_asked(self, candidate: p.Parameter, value: float) -> None:
        # nearly same as DE
        candidate._meta["loss"] = value
        worst: tp.Optional[p.Parameter] = None
        if not len(self.population) < self.llambda:
            worst = max(self.population.values(), key=lambda p: p._meta.get("value", float("inf")))
            if worst._meta.get("value", float("inf")) < value:
                return  # no need to update
            else:
                uid = worst.heritage["lineage"]
                del self.population[uid]
                self._uid_queue.discard(uid)
        candidate.heritage["lineage"] = candidate.uid  # new lineage
        if "speed" not in candidate.heritage:
            candidate.heritage["speed"] = self._rng.uniform(-1.0, 1.0, self.parametrization.dimension)
        self.population[candidate.uid] = candidate
        self._uid_queue.tell(candidate.uid)
        if value < self._best._meta.get("loss", float("inf")):
            self._best = candidate


class ConfiguredPSO(base.ConfiguredOptimizer):

    # pylint: disable=unused-argument
    def __init__(self, transform: str = "identity", wide: bool = False) -> None:
        super().__init__(PSO, locals())


RealSpacePSO = ConfiguredPSO().set_name("RealSpacePSO", register=True)


@registry.register
class SPSA(base.Optimizer):
    # pylint: disable=too-many-instance-attributes
    """ The First order SPSA algorithm as shown in [1,2,3], with implementation details
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

    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.init = True
        self.idx = 0
        self.delta = float("nan")
        self.ym: Optional[np.ndarray] = None
        self.yp: Optional[np.ndarray] = None
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
        # the magnitude of gradient and propotional to (1+A)^0.602
        # we should spend some burn-in iterations to estimate the
        # magnitude of the gradient. 1e-5 is arbitrary.
        self.a = 1e-5

    def _ck(self, k: int) -> float:
        "c_k determines the pertubation."
        return self.c / (k // 2 + 1) ** 0.101

    def _ak(self, k: int) -> float:
        "a_k is the learning rate."
        return self.a / (k // 2 + 1 + self.A) ** 0.602

    def _internal_ask(self) -> ArrayLike:
        k = self.idx
        if k % 2 == 0:
            if not self.init:
                assert self.yp is not None and self.ym is not None
                self.t -= (self._ak(k) * (self.yp - self.ym) / 2 / self._ck(k)) * self.delta
                self.avg += (self.t - self.avg) / (k // 2 + 1)
            self.delta = 2 * self._rng.randint(2, size=self.dimension) - 1
            return self.t - self._ck(k) * self.delta  # type:ignore
        return self.t + self._ck(k) * self.delta  # type: ignore

    def _internal_tell(self, x: ArrayLike, value: float) -> None:
        setattr(self, ("ym" if self.idx % 2 == 0 else "yp"), np.array(value, copy=True))
        self.idx += 1
        if self.init and self.yp is not None and self.ym is not None:
            self.init = False

    def _internal_provide_recommendation(self) -> ArrayLike:
        return self.avg


@registry.register
class SplitOptimizer(base.Optimizer):
    """Combines optimizers, each of them working on their own variables.

    num_optims: number of optimizers
    num_vars: number of variable per optimizer.
    progressive: True if we want to progressively add optimizers during the optimization run.

    If progressive = True, the optimizer is forced at OptimisticNoisyOnePlusOne.

    E.g. for 5 optimizers, each of them working on 2 variables, we can use:
    opt = SplitOptimizer(parametrization=10, num_workers=3, num_optims=5, num_vars=[2, 2, 2, 2, 2])
    or equivalently:
    opt = SplitOptimizer(parametrization=10, num_workers=3, num_vars=[2, 2, 2, 2, 2])
    Given that all optimizers have the same number of variables, we can also do:
    opt = SplitOptimizer(parametrization=10, num_workers=3, num_optims=5)

    This is 5 parallel (by num_workers = 5).

    Be careful! The variables refer to the deep representation used by optimizers.
    For example, a categorical variable with 5 possible values becomes 5 continuous variables.
    """

    def __init__(
            self,
            parametrization: IntOrParameter,
            budget: Optional[int] = None,
            num_workers: int = 1,
            num_optims: tp.Optional[int] = None,
            num_vars: Optional[List[int]] = None,
            multivariate_optimizer: base.ConfiguredOptimizer = CMA,
            monovariate_optimizer: base.ConfiguredOptimizer = RandomSearch,
            progressive: bool = False,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        if num_vars is not None:
            if num_optims is not None:
                assert num_optims == len(num_vars), f"The number {num_optims} of optimizers should match len(num_vars)={len(num_vars)}."
            else:
                num_optims = len(num_vars)
            assert sum(num_vars) == self.dimension, f"sum(num_vars)={sum(num_vars)} should be equal to the dimension {self.dimension}."
        else:
            if num_optims is None:  # if no num_vars and no num_optims, just assume 2.
                num_optims = 2
            # if num_vars not given: we will distribute variables equally.
        if num_optims > self.dimension:
            num_optims = self.dimension
        self.num_optims = num_optims
        self.progressive = progressive
        self.optims: List[Any] = []
        self.num_vars: List[Any] = num_vars if num_vars else []
        self.parametrizations: List[Any] = []
        for i in range(self.num_optims):
            if not self.num_vars or len(self.num_vars) < i + 1:
                self.num_vars += [(self.dimension // self.num_optims) + (self.dimension % self.num_optims > i)]

            assert self.num_vars[i] >= 1, "At least one variable per optimizer."
            self.parametrizations += [p.Array(shape=(self.num_vars[i],))]
            for param in self.parametrizations:
                param.random_state = self.parametrization.random_state
            assert len(self.optims) == i
            if self.num_vars[i] > 1:
                self.optims += [multivariate_optimizer(self.parametrizations[i], budget, num_workers)]  # noqa: F405
            else:
                self.optims += [monovariate_optimizer(self.parametrizations[i], budget, num_workers)]  # noqa: F405

        assert sum(
            self.num_vars) == self.dimension, f"sum(num_vars)={sum(self.num_vars)} should be equal to the dimension {self.dimension}."

    def _internal_ask_candidate(self) -> p.Parameter:
        data: List[Any] = []
        for i in range(self.num_optims):
            if self.progressive:
                assert self.budget is not None
                if i > 0 and i / self.num_optims > np.sqrt(2.0 * self._num_ask / self.budget):
                    data += [0.] * self.num_vars[i]
                    continue
            opt = self.optims[i]
            data += list(opt.ask().get_standardized_data(reference=opt.parametrization))
        assert len(data) == self.dimension
        return self.parametrization.spawn_child().set_standardized_data(data)

    def _internal_tell_candidate(self, candidate: p.Parameter, value: float) -> None:
        data = candidate.get_standardized_data(reference=self.parametrization)
        n = 0
        for i in range(self.num_optims):
            opt = self.optims[i]
            local_data = list(data)[n:n + self.num_vars[i]]
            n += self.num_vars[i]
            assert len(local_data) == self.num_vars[i]
            local_candidate = opt.parametrization.spawn_child().set_standardized_data(local_data)
            opt.tell(local_candidate, value)

    def _internal_provide_recommendation(self) -> ArrayLike:
        return self.current_bests["pessimistic"].x

    def _internal_tell_not_asked(self, candidate: p.Parameter, value: float) -> None:
        raise base.TellNotAskedNotSupportedError


class ConfSplitOptimizer(base.ConfiguredOptimizer):
    """Configurable split optimizer

    Parameters
    ----------
    num_optims: int
        number of optimizers
    num_vars: optional list of int
        number of variable per optimizer.
    progressive: optional bool
        whether we progressively add optimizers.
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        *,
        num_optims: int = 2,
        num_vars: tp.Optional[tp.List[int]] = None,
        multivariate_optimizer: base.ConfiguredOptimizer = CMA,
        monovariate_optimizer: base.ConfiguredOptimizer = RandomSearch,
        progressive: bool = False
    ) -> None:
        if progressive:  # The progressive setting is typically for noisy optimization, hence we switch to a noisy optimizer.
            multivariate_optimizer = OptimisticNoisyOnePlusOne
        super().__init__(SplitOptimizer, locals())


@registry.register
class Portfolio(base.Optimizer):
    """Passive portfolio of CMA, 2-pt DE and Scr-Hammersley."""

    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        assert budget is not None
        self.optims = [
            CMA(self.parametrization, budget // 3 + (budget % 3 > 0), num_workers),  # share parametrization and its rng
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

    def _internal_tell_candidate(self, candidate: p.Parameter, value: float) -> None:
        optim_index: int = candidate._meta["optim_index"]
        self.optims[optim_index].tell(candidate, value)

    def _internal_provide_recommendation(self) -> ArrayLike:
        return self.current_bests["pessimistic"].x

    def _internal_tell_not_asked(self, candidate: p.Parameter, value: float) -> None:
        raise base.TellNotAskedNotSupportedError


@registry.register
class ParaPortfolio(Portfolio):
    """Passive portfolio of CMA, 2-pt DE, PSO, SQP and Scr-Hammersley."""

    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        assert budget is not None

        def intshare(n: int, m: int) -> Tuple[int, ...]:
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
        self.optims: List[base.Optimizer] = [
            CMA(self.parametrization, num_workers=nw1),  # share parametrization and its rng
            TwoPointsDE(self.parametrization, num_workers=nw2),  # noqa: F405
            PSO(self.parametrization, num_workers=nw3),
            SQP(self.parametrization, 1),  # noqa: F405
            ScrHammersleySearch(self.parametrization, budget=(budget // len(self.which_optim)) * nw4),  # noqa: F405
        ]

    def _internal_ask_candidate(self) -> p.Parameter:
        optim_index = self.which_optim[self._num_ask % len(self.which_optim)]
        candidate = self.optims[optim_index].ask()
        candidate._meta["optim_index"] = optim_index
        return candidate


@registry.register
class SQPCMA(ParaPortfolio):
    """Passive portfolio of CMA and many SQP."""

    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
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
            self.optims += [SQP(self.parametrization, 1)]  # noqa: F405
            if i > 0:
                self.optims[-1].initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore


@registry.register
class ASCMADEthird(Portfolio):
    """Algorithm selection, with CMA and Lhs-DE. Active selection at 1/3."""

    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        assert budget is not None
        self.optims = [
            CMA(self.parametrization, budget=None, num_workers=num_workers),  # share parametrization and its rng
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
                best_value = float("inf")
                optim_index = -1
                for i, optim in enumerate(self.optims):
                    val = optim.current_bests["pessimistic"].get_estimation("pessimistic")
                    if not val > best_value:
                        optim_index = i
                        best_value = val
                self.best_optim = optim_index
            optim_index = self.best_optim
        candidate = self.optims[optim_index].ask()
        candidate._meta["optim_index"] = optim_index
        return candidate


@registry.register
class ASCMADEQRthird(ASCMADEthird):
    """Algorithm selection, with CMA, ScrHalton and Lhs-DE. Active selection at 1/3."""

    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.optims = [
            CMA(self.parametrization, budget=None, num_workers=num_workers),
            LhsDE(self.parametrization, budget=None, num_workers=num_workers),  # noqa: F405
            ScrHaltonSearch(self.parametrization, budget=None, num_workers=num_workers),
        ]  # noqa: F405


@registry.register
class ASCMA2PDEthird(ASCMADEQRthird):
    """Algorithm selection, with CMA and 2pt-DE. Active selection at 1/3."""

    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.optims = [
            CMA(self.parametrization, budget=None, num_workers=num_workers),
            TwoPointsDE(self.parametrization, budget=None, num_workers=num_workers),
        ]  # noqa: F405


@registry.register
class CMandAS2(ASCMADEthird):
    """Competence map, with algorithm selection in one of the cases (3 CMAs)."""

    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.optims = [TwoPointsDE(self.parametrization, budget=None, num_workers=num_workers)]  # noqa: F405
        assert budget is not None
        self.budget_before_choosing = 2 * budget
        if budget < 201:
            self.optims = [OnePlusOne(self.parametrization, budget=None, num_workers=num_workers)]
        if budget > 50 * self.dimension or num_workers < 30:
            self.optims = [
                CMA(self.parametrization, budget=None, num_workers=num_workers),  # share parametrization and its rng
                CMA(self.parametrization, budget=None, num_workers=num_workers),
                CMA(self.parametrization, budget=None, num_workers=num_workers),
            ]
            self.budget_before_choosing = budget // 10


@registry.register
class CMandAS3(ASCMADEthird):
    """Competence map, with algorithm selection in one of the cases (3 CMAs)."""

    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.optims = [TwoPointsDE(self.parametrization, budget=None, num_workers=num_workers)]  # noqa: F405
        assert budget is not None
        self.budget_before_choosing = 2 * budget
        if budget < 201:
            self.optims = [OnePlusOne(self.parametrization, budget=None, num_workers=num_workers)]
        if budget > 50 * self.dimension or num_workers < 30:
            if num_workers == 1:
                self.optims = [
                    chainCMAPowell(self.parametrization, budget=None, num_workers=num_workers),  # share parametrization and its rng
                    chainCMAPowell(self.parametrization, budget=None, num_workers=num_workers),
                    chainCMAPowell(self.parametrization, budget=None, num_workers=num_workers),
                ]
            else:
                self.optims = [
                    CMA(self.parametrization, budget=None, num_workers=num_workers),  # share parametrization and its rng
                    CMA(self.parametrization, budget=None, num_workers=num_workers),
                    CMA(self.parametrization, budget=None, num_workers=num_workers),
                ]
            self.budget_before_choosing = budget // 10


@registry.register
class CMandAS(CMandAS2):
    """Competence map, with algorithm selection in one of the cases (2 CMAs)."""

    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.optims = [TwoPointsDE(self.parametrization, budget=None, num_workers=num_workers)]  # noqa: F405
        assert budget is not None
        self.budget_before_choosing = 2 * budget
        if budget < 201:
            # share parametrization and its rng
            self.optims = [OnePlusOne(self.parametrization, budget=None, num_workers=num_workers)]
            self.budget_before_choosing = 2 * budget
        if budget > 50 * self.dimension or num_workers < 30:
            self.optims = [
                CMA(self.parametrization, budget=None, num_workers=num_workers),
                CMA(self.parametrization, budget=None, num_workers=num_workers),
            ]
            self.budget_before_choosing = budget // 3


@registry.register
class CM(CMandAS2):
    """Competence map, simplest."""

    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
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

    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        assert budget is not None
        self.optims = [
            CMA(self.parametrization, budget=None, num_workers=num_workers),  # share parametrization and its rng
            CMA(self.parametrization, budget=None, num_workers=num_workers),
            CMA(self.parametrization, budget=None, num_workers=num_workers),
        ]
        self.budget_before_choosing = budget // 10


@registry.register
class TripleCMA(CM):
    """Combining 3 CMAs. Exactly identical. Active selection at 1/3 of the budget."""

    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        assert budget is not None
        self.optims = [
            CMA(self.parametrization, budget=None, num_workers=num_workers),  # share parametrization and its rng
            CMA(self.parametrization, budget=None, num_workers=num_workers),
            CMA(self.parametrization, budget=None, num_workers=num_workers),
        ]
        self.budget_before_choosing = budget // 3


@registry.register
class MultiScaleCMA(CM):
    """Combining 3 CMAs with different init scale. Active selection at 1/3 of the budget."""

    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.optims = [
            CMA(self.parametrization, budget=None, num_workers=num_workers),  # share parametrization and its rng
            ParametrizedCMA(scale=1e-3)(self.parametrization, budget=None, num_workers=num_workers),
            ParametrizedCMA(scale=1e-6)(self.parametrization, budget=None, num_workers=num_workers),
        ]
        assert budget is not None
        self.budget_before_choosing = budget // 3


class _FakeFunction:
    """Simple function that returns the value which was registerd just before.
    This is a hack for BO.
    """

    def __init__(self) -> None:
        self._registered: List[Tuple[np.ndarray, float]] = []

    def register(self, x: np.ndarray, value: float) -> None:
        if self._registered:
            raise RuntimeError("Only one call can be registered at a time")
        self._registered.append((x, value))

    def __call__(self, **kwargs: float) -> float:
        if not self._registered:
            raise RuntimeError("Call must be registered first")
        x = [kwargs[f"x{i}"] for i in range(len(kwargs))]
        xr, value = self._registered[0]
        if not np.array_equal(x, xr):
            raise ValueError("Call does not match registered")
        self._registered.clear()
        return value


class _BO(base.Optimizer):

    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: Optional[int] = None,
        num_workers: int = 1,
        *,
        initialization: Optional[str] = None,
        init_budget: Optional[int] = None,
        middle_point: bool = False,
        utility_kind: str = "ucb",  # bayes_opt default
        utility_kappa: float = 2.576,
        utility_xi: float = 0.0,
        gp_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._transform = transforms.ArctanBound(0, 1)
        self._bo: Optional[BayesianOptimization] = None
        self._fake_function = _FakeFunction()
        # configuration
        assert initialization is None or initialization in ["random", "Hammersley", "LHS"], f"Unknown init {initialization}"
        self.initialization = initialization
        self.init_budget = init_budget
        self.middle_point = middle_point
        self.utility_kind = utility_kind
        self.utility_kappa = utility_kappa
        self.utility_xi = utility_xi
        self.gp_parameters = {} if gp_parameters is None else gp_parameters
        if isinstance(parametrization, p.Parameter) and self.gp_parameters.get("alpha", 0) == 0:
            noisy = not parametrization.descriptors.deterministic
            cont = parametrization.descriptors.continuous
            if noisy or not cont:
                warnings.warn(
                    "Dis-continuous and noisy parametrization require gp_parameters['alpha'] > 0 "
                    "(for your parametrization, continuity={cont} and noisy={noisy}).\n"
                    "Find more information on BayesianOptimization's github.\n"
                    "You should then create a new instance of optimizerlib.ParametrizedBO with appropriate parametrization.",
                    InefficientSettingsWarning,
                )

    @property
    def bo(self) -> BayesianOptimization:
        if self._bo is None:
            bounds = {f"x{i}": (0.0, 1.0) for i in range(self.dimension)}
            self._bo = BayesianOptimization(self._fake_function, bounds, random_state=self._rng)
            if self.gp_parameters is not None:
                self._bo.set_gp_params(**self.gp_parameters)
            # init
            init = self.initialization
            if self.middle_point:
                self._bo.probe([0.5] * self.dimension, lazy=True)
            elif init is None:
                self._bo._queue.add(self._bo._space.random_sample())
            if init is not None:
                init_budget = int(np.sqrt(self.budget) if self.init_budget is None else self.init_budget)
                init_budget -= self.middle_point
                if init_budget > 0:
                    sampler = {"Hammersley": sequences.HammersleySampler, "LHS": sequences.LHSSampler, "random": sequences.RandomSampler}[
                        init
                    ](self.dimension, budget=init_budget, scrambling=(init == "Hammersley"), random_state=self._rng)
                    for point in sampler:
                        self._bo.probe(point, lazy=True)
        return self._bo

    def _internal_ask_candidate(self) -> p.Parameter:
        util = UtilityFunction(kind=self.utility_kind, kappa=self.utility_kappa, xi=self.utility_xi)
        try:
            x_probe = next(self.bo._queue)
        except StopIteration:
            x_probe = self.bo.suggest(util)  # this is time consuming
            x_probe = [x_probe[f"x{i}"] for i in range(len(x_probe))]
        data = self._transform.backward(np.array(x_probe, copy=False))
        candidate = self.parametrization.spawn_child().set_standardized_data(data)
        candidate._meta["x_probe"] = x_probe
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, value: float) -> None:
        if "x_probe" in candidate._meta:
            y = candidate._meta["x_probe"]
        else:
            data = candidate.get_standardized_data(reference=self.parametrization)
            y = self._transform.forward(data)  # tell not asked
        self._fake_function.register(y, -value)  # minimizing
        self.bo.probe(y, lazy=False)
        # for some unknown reasons, BO wants to evaluate twice the same point,
        # but since it keeps a cache of the values, the registered value is not used
        # so we should clean the "fake" function
        self._fake_function._registered.clear()

    def _internal_provide_recommendation(self) -> ArrayLike:
        if self.archive:
            return self._transform.backward(np.array([self.bo.max["params"][f"x{i}"] for i in range(self.dimension)]))
        else:
            return super()._internal_provide_recommendation()


class ParametrizedBO(base.ConfiguredOptimizer):
    """Bayesian optimization

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
        initialization: Optional[str] = None,
        init_budget: Optional[int] = None,
        middle_point: bool = False,
        utility_kind: str = "ucb",  # bayes_opt default
        utility_kappa: float = 2.576,
        utility_xi: float = 0.0,
        gp_parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(_BO, locals())


BO = ParametrizedBO().set_name("BO", register=True)


@registry.register
class PBIL(base.Optimizer):
    """
    Implementation of the discrete algorithm PBIL

    https://www.ri.cmu.edu/pub_files/pub1/baluja_shumeet_1994_2/baluja_shumeet_1994_2.pdf
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)

        self._penalize_cheap_violations = False  # Not sure this is the optimal decision.
        num_categories = 2
        self.p: np.ndarray = np.ones((1, self.dimension)) / num_categories
        self.alpha = 0.3
        self.llambda = max(100, num_workers)  # size of the population
        self.mu = self.llambda // 2  # number of selected candidates
        self._population: List[Tuple[float, np.ndarray]] = []

    def _internal_ask_candidate(self) -> p.Parameter:
        unif = self._rng.uniform(size=self.dimension)
        data = (unif > 1 - self.p[0]).astype(float)
        return self.parametrization.spawn_child().set_standardized_data(data)

    def _internal_tell_candidate(self, candidate: p.Parameter, value: float) -> None:
        data = candidate.get_standardized_data(reference=self.parametrization)
        self._population.append((value, data))
        if len(self._population) >= self.llambda:
            self._population.sort(key=lambda tup: tup[0])
            mean_pop: np.ndarray = np.mean([x[1] for x in self._population[: self.mu]])
            self.p[0] = (1 - self.alpha) * self.p[0] + self.alpha * mean_pop
            self._population = []


class _Chain(base.Optimizer):

    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: Optional[int] = None,
        num_workers: int = 1,
        *,
        optimizers: tp.Sequence[tp.Union[base.ConfiguredOptimizer, tp.Type[base.Optimizer]]] = [LHSSearch, DE],
        budgets: tp.Sequence[tp.Union[str, int]] = (10,),
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        # delayed initialization
        # Either we have the budget for each algorithm, or the last algorithm uses the rest of the budget, so:
        self.optimizers: tp.List[base.Optimizer] = []
        converter = {"num_workers": self.num_workers, "dimension": self.dimension,
                     "half": self.budget // 2 if self.budget else self.num_workers,
                     "sqrt": int(np.sqrt(self.budget)) if self.budget else self.num_workers}
        self.budgets = [converter[b] if isinstance(b, str) else b for b in budgets]
        last_budget = None if self.budget is None else self.budget - sum(self.budgets)
        assert len(optimizers) == len(self.budgets) + 1
        assert all(x in ("half", "dimension", "num_workers", "sqrt") or x > 0 for x in self.budgets)
        for opt, optbudget in zip(optimizers, self.budgets + [last_budget]):  # type: ignore
            self.optimizers.append(opt(self.parametrization, budget=optbudget, num_workers=self.num_workers))

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

    def _internal_tell_candidate(self, candidate: p.Parameter, value: float) -> None:
        # Let us inform all concerned algorithms
        sum_budget = 0.0
        for opt in self.optimizers:
            sum_budget += float("inf") if opt.budget is None else opt.budget
            if self.num_tell < sum_budget:
                opt.tell(candidate, value)


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
        budgets: tp.Sequence[tp.Union[str, int]]
    ) -> None:
        super().__init__(_Chain, locals())


chainCMAPowell = Chaining([CMA, Powell], ["half"]).set_name("chainCMAPowell", register=True)
chainCMAPowell.no_parallelization = True


@registry.register
class cGA(base.Optimizer):
    """
    Implementation of the discrete cGA algorithm

    https://pdfs.semanticscholar.org/4b0b/5733894ffc0b2968ddaab15d61751b87847a.pdf
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: Optional[int] = None,
        num_workers: int = 1,
        arity: Optional[int] = None
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        if arity is None:
            arity = len(parametrization.possibilities) if hasattr(parametrization, "possibilities") else 2  # type: ignore
        self._arity = arity
        self._penalize_cheap_violations = False  # Not sure this is the optimal decision.
        # self.p[i][j] is the probability that the ith variable has value 0<=j< arity.
        self.p: np.ndarray = np.ones((self.dimension, arity)) / arity
        # Probability increments are of order 1./self.llambda
        # and lower bounded by something of order 1./self.llambda.
        self.llambda = max(num_workers, 40)  # FIXME: no good heuristic ?
        # CGA generates a candidate, then a second candidate;
        # then updates depending on the comparison with the first one. We therefore have to store the previous candidate.
        self._previous_value_candidate: Optional[Tuple[float, np.ndarray]] = None

    def _internal_ask_candidate(self) -> p.Parameter:
        # Multinomial.
        values: List[int] = [sum(self._rng.uniform() > cum_proba) for cum_proba in np.cumsum(self.p, axis=1)]
        data = discretization.noisy_inverse_threshold_discretization(values, arity=self._arity, gen=self._rng)
        return self.parametrization.spawn_child().set_standardized_data(data)

    def _internal_tell_candidate(self, candidate: p.Parameter, value: float) -> None:
        data = candidate.get_standardized_data(reference=self.parametrization)
        if self._previous_value_candidate is None:
            self._previous_value_candidate = (value, data)
        else:
            winner, loser = self._previous_value_candidate[1], data
            if self._previous_value_candidate[0] > value:
                winner, loser = loser, winner
            winner_data = discretization.threshold_discretization(np.asarray(winner.data), arity=self._arity)
            loser_data = discretization.threshold_discretization(np.asarray(loser.data), arity=self._arity)
            for i, _ in enumerate(winner_data):
                if winner_data[i] != loser_data[i]:
                    self.p[i][winner_data[i]] += 1. / self.llambda
                    self.p[i][loser_data[i]] -= 1. / self.llambda
                    for j in range(len(self.p[i])):
                        self.p[i][j] = max(self.p[i][j], 1. / self.llambda)
                    self.p[i] /= sum(self.p[i])
            self._previous_value_candidate = None


# Discussions with Jialin Liu and Fabien Teytaud helped the following development.
# This includes discussion at Dagstuhl's 2019 seminars on randomized search heuristics and computational intelligence in games.
@registry.register
class NGO(base.Optimizer):
    """Nevergrad optimizer by competence map."""
    one_shot = True

    # pylint: disable=too-many-branches
    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        assert budget is not None
        descr = self.parametrization.descriptors
        self.has_noise = not (descr.deterministic and descr.deterministic_function)
        self.fully_continuous = descr.continuous
        all_params = paramhelpers.flatten_parameter(self.parametrization)
        self.has_discrete_not_softmax = any(isinstance(x, p.TransitionChoice) for x in all_params.values())
        # pylint: disable=too-many-nested-blocks
        if self.has_noise and self.has_discrete_not_softmax:
            # noise and discrete: let us merge evolution and bandits.
            if self.dimension < 60:
                self.optims = [DoubleFastGADiscreteOnePlusOne(self.parametrization, budget, num_workers)]
            else:
                self.optims = [CMA(self.parametrization, budget, num_workers)]
        else:
            if self.has_noise and self.fully_continuous:
                # This is the real of population control. FIXME: should we pair with a bandit ?
                self.optims = [TBPSA(self.parametrization, budget, num_workers)]
            else:
                if self.has_discrete_not_softmax or not self.parametrization.descriptors.metrizable or not self.fully_continuous:
                    self.optims = [DoubleFastGADiscreteOnePlusOne(self.parametrization, budget, num_workers)]
                else:
                    if num_workers > budget / 5:
                        if num_workers > budget / 2. or budget < self.dimension:
                            self.optims = [MetaRecentering(self.parametrization, budget, num_workers)]  # noqa: F405
                        else:
                            self.optims = [NaiveTBPSA(self.parametrization, budget, num_workers)]  # noqa: F405
                    else:
                        # Possibly a good idea to go memetic for large budget, but something goes wrong for the moment.
                        if num_workers == 1 and budget > 6000 and self.dimension > 7:  # Let us go memetic.
                            self.optims = [chainCMAPowell(self.parametrization, budget, num_workers)]  # noqa: F405
                        else:
                            if num_workers == 1 and budget < self.dimension * 30:
                                if self.dimension > 30:  # One plus one so good in large ratio "dimension / budget".
                                    self.optims = [OnePlusOne(self.parametrization, budget, num_workers)]  # noqa: F405
                                else:
                                    self.optims = [Cobyla(self.parametrization, budget, num_workers)]  # noqa: F405
                            else:
                                if self.dimension > 2000:  # DE is great in such a case (?).
                                    self.optims = [DE(self.parametrization, budget, num_workers)]  # noqa: F405
                                else:
                                    self.optims = [CMA(self.parametrization, budget, num_workers)]  # noqa: F405

    def _internal_ask_candidate(self) -> p.Parameter:
        optim_index = 0
        candidate = self.optims[optim_index].ask()
        candidate._meta["optim_index"] = optim_index
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, value: float) -> None:
        optim_index = candidate._meta["optim_index"]
        self.optims[optim_index].tell(candidate, value)

    def _internal_provide_recommendation(self) -> ArrayLike:
        params = self.optims[0].provide_recommendation()
        return params.get_standardized_data(reference=self.parametrization)

    def _internal_tell_not_asked(self, candidate: p.Parameter, value: float) -> None:
        raise base.TellNotAskedNotSupportedError


class _EMNA(base.Optimizer):
    """Simple Estimation of Multivariate Normal Algorithm (EMNA).
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(
            self,
            parametrization: IntOrParameter,
            budget: Optional[int] = None,
            num_workers: int = 1,
            isotropic: bool = True,
            naive: bool = True
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.isotropic: bool = isotropic
        self.naive: bool = naive
        self.sigma: tp.Union[float, np.ndarray]
        if self.isotropic:
            self.sigma = 1.0
        else:
            self.sigma = np.ones(self.dimension)
        self.mu = max(16, self.dimension)
        self.llambda = 4 * self.mu
        if budget is not None and self.llambda > budget:
            self.llambda = budget
            self.mu = self.llambda // 4
            warnings.warn("Budget may be too small in front of the dimension for EMNA")
        if num_workers is not None:
            self.llambda = max(self.llambda, num_workers)
        self.current_center: np.ndarray = np.zeros(self.dimension)
        # population
        self.parents: List[p.Parameter] = [self.parametrization]
        self.children: List[p.Parameter] = []

    def _internal_provide_recommendation(self) -> ArrayLike:
        if self.naive:
            return self.current_bests["optimistic"].x
        else:
            return self.current_center

    def _internal_ask_candidate(self) -> p.Parameter:
        individual = self.current_center + self.sigma * self._rng.normal(0, 1, self.dimension)
        parent = self.parents[self.num_ask % len(self.parents)]
        candidate = parent.spawn_child().set_standardized_data(individual, reference=self.parametrization)
        if parent is self.parametrization:
            candidate.heritage["lineage"] = candidate.uid
        candidate._meta["sigma"] = self.sigma
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, value: float) -> None:
        candidate._meta["loss"] = value
        self.children.append(candidate)
        if len(self.children) >= self.llambda:
            # Sorting the population.
            self.children.sort(key=lambda c: c._meta["loss"])
            # Computing the new parent.
            self.parents = self.children[: self.mu]
            self.children = []
            self.current_center = sum(c.get_standardized_data(reference=self.parametrization)  # type: ignore
                                      for c in self.parents) / self.mu
            # EMNA update
            stdd = [(self.parents[i].get_standardized_data(reference=self.parametrization) - self.current_center)**2
                    for i in range(self.mu)]
            if self.isotropic:
                self.sigma = np.sqrt(sum(stdd) / (self.mu * self.dimension))
            else:
                self.sigma = np.sqrt(np.sum(stdd, axis=0) / (self.mu))

            if self.num_workers / self.dimension > 32:  # faster decrease of sigma if large parallel context
                imp = max(1, (np.log(self.llambda) / 2)**(1 / self.dimension))
                self.sigma /= imp

    def _internal_tell_not_asked(self, candidate: p.Parameter, value: float) -> None:
        raise base.TellNotAskedNotSupportedError


class EMNA(base.ConfiguredOptimizer):
    """ Estimation of Multivariate Normal Algorithm
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
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        *,
        isotropic: bool = True,
        naive: bool = True
    ) -> None:
        super().__init__(_EMNA, locals())


NaiveIsoEMNA = EMNA().set_name("NaiveIsoEMNA", register=True)


@registry.register
class Shiva(NGO):
    """Nevergrad optimizer by competence map. You might modify this one for designing youe own competence map."""

    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        assert budget is not None
        if self.has_noise and (self.has_discrete_not_softmax or not self.parametrization.descriptors.metrizable):
            self.optims = [RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne(self.parametrization, budget, num_workers)]
        else:
            if not self.parametrization.descriptors.metrizable:
                if self.dimension < 60:
                    self.optims = [NGO(self.parametrization, budget, num_workers)]
                else:
                    self.optims = [CMA(self.parametrization, budget, num_workers)]
            else:
                self.optims = [NGO(self.parametrization, budget, num_workers)]
