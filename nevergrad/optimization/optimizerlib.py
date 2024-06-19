# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import logging
import itertools
from collections import deque
import warnings
import numpy as np
from collections import defaultdict
import scipy.ndimage as ndimage

try:
    from bayes_opt import UtilityFunction
    from bayes_opt import BayesianOptimization
except ModuleNotFoundError:
    pass

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
from .metamodel import MetaModelFailure as MetaModelFailure
from .metamodel import learn_on_k_best as learn_on_k_best
from .base import registry as registry
from .base import addCompare  # pylint: disable=unused-import
from .base import IntOrParameter

# families of optimizers
# pylint: disable=unused-wildcard-import,wildcard-import,too-many-lines,too-many-arguments,too-many-branches
# pylint: disable=import-outside-toplevel,too-many-nested-blocks,too-many-instance-attributes,
# pylint: disable=too-many-boolean-expressions,too-many-ancestors,too-many-statements,too-many-return-statements
from .differentialevolution import *  # type: ignore  # noqa: F403
from .es import *  # type: ignore  # noqa: F403
from .oneshot import *  # noqa: F403
from .recastlib import *  # noqa: F403

try:
    from .externalbo import HyperOpt  # pylint: disable=unused-import

    # HyperOpt = ParametrizedHyperOpt().set_name("HyperOpt", register=True)
except:  # ModuleNotFoundError:
    pass

# from .externalbo import ParametrizedHyperOpt
# Lamcts = LamctsOptimizer(random_restart=True, device="cpu").set_name("Lamcts", register=True)
# Lamcts2 = LamctsOptimizer(random_restart=True, device="cpu").set_name("Lamcts2", register=True)

# run with LOGLEVEL=DEBUG for more debug information
# logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


# # # # # optimizers # # # # #


def smooth_copy(array: p.Array, possible_radii: tp.Optional[tp.List[int]] = None) -> p.Array:
    candidate = array.spawn_child()
    if possible_radii is None:
        possible_radii = [3]
    value = candidate._value
    radii = [array.random_state.choice(possible_radii) for _ in value.shape]
    value2 = ndimage.convolve(value, np.ones(radii) / np.prod(radii))
    # DE style operator.
    indices = array.random_state.randint(4, size=value.shape) == 0
    value[indices] = value2[indices]
    candidate._value = value
    return candidate


class _OnePlusOne(base.Optimizer):
    """Simple but sometimes powerful optimization algorithm.

    We use the one-fifth adaptation rule, going back to Schumer and Steiglitz (1968).
    It was independently rediscovered by Devroye (1972) and Rechenberg (1973).
    We use asynchronous updates, so that the 1+1 can actually be parallel and even
    performs quite well in such a context - this is naturally close to 1+lambda.

    Posssible mutations include gaussian and cauchy for the continuous case, and in the discrete case:
    discrete, fastga, rls, doublefastga, adaptive, portfolio, discreteBSO, doerr.
    - discrete is the most classical discrete mutation operator,
    - rls is the Randomized Local Search,
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
        tabu_length: int = 0,
        mutation: str = "gaussian",
        crossover: bool = False,
        rotation: bool = False,
        annealing: str = "none",
        use_pareto: bool = False,
        sparse: tp.Union[bool, int] = False,
        smoother: bool = False,
        roulette_size: int = 2,
        antismooth: int = 55,
        crossover_type: str = "none",
        forced_discretization: bool = False,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.parametrization.tabu_length = tabu_length
        if forced_discretization:
            self.parametrization.set_integer_casting()  # type: ignore
        self.antismooth = antismooth
        self.crossover_type = crossover_type
        self.roulette_size = roulette_size
        assert crossover or (not rotation), "We can not have both rotation and not crossover."
        self._sigma: float = 1
        self._previous_best_loss = float("inf")
        self._best_recent_mr = 0.2
        self.inds = np.array([True] * self.dimension)
        self.imr = 0.2
        self.use_pareto = use_pareto
        self.smoother = smoother
        self.annealing = annealing
        self._annealing_base: tp.Optional[tp.ArrayLike] = None
        self._max_loss = -float("inf")
        self.sparse = int(sparse)  # True --> 1
        all_params = p.helpers.flatten(self.parametrization)
        arities = [len(param.choices) for _, param in all_params if isinstance(param, p.TransitionChoice)]
        arity = max(arities, default=500)
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
            "rls",
            "doublefastga",
            "adaptive",
            "coordinatewise_adaptive",
            "portfolio",
            "discreteBSO",
            "lengler",
            "lengler2",
            "lengler3",
            "lenglerhalf",
            "lenglerfourth",
            "doerr",
            "lognormal",
            "xlognormal",
            "xsmalllognormal",
            "lognormal",
            "smalllognormal",
            "biglognormal",
            "hugelognormal",
        ], f"Unkwnown mutation: '{mutation}'"
        if mutation == "adaptive":
            self._adaptive_mr = 0.5
        elif mutation == "lognormal":
            self._global_mr = 0.2
            self._memory_index = 0
            self._memory_size = 12  # Dirty random value
            self._best_recent_loss = float("inf")
        elif mutation == "xsmalllognormal":
            self._global_mr = 0.8
            self._memory_index = 0
            self._memory_size = 4  # Dirty random value
            self._best_recent_loss = float("inf")
        elif mutation == "xlognormal":
            self._global_mr = 0.8
            self._memory_index = 0
            self._memory_size = 12  # Dirty random value
            self._best_recent_loss = float("inf")
        elif mutation == "smalllognormal":
            self._global_mr = 0.2
            self._memory_index = 0
            self._memory_size = 4  # Dirty random value
            self._best_recent_loss = float("inf")
        elif mutation == "biglognormal":
            self._global_mr = 0.2
            self._memory_index = 0
            self._memory_size = 120  # Dirty random value
            self._best_recent_loss = float("inf")
        elif mutation == "hugelognormal":
            self._global_mr = 0.2
            self._memory_index = 0
            self._memory_size = 1200  # Dirty random value
            self._best_recent_loss = float("inf")
        elif mutation == "coordinatewise_adaptive":
            self._velocity = self._rng.uniform(size=self.dimension) * arity / 4.0
            self._modified_variables = np.array([True] * self.dimension)
        self.noise_handling = noise_handling
        self.mutation = mutation
        self.crossover = crossover
        self.rotation = rotation
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
        assert self.parametrization.tabu_length == tabu_length

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
        if (
            self.smoother
            and self._num_ask % max(self.num_workers + 1, self.antismooth) == 0
            and isinstance(self.parametrization, p.Array)
        ):
            self.suggest(smooth_copy(pessimistic).value)  # type: ignore
        if self.num_objectives > 1 and self.use_pareto:  # multiobjective
            # revert to using a sample of the pareto front (not "pessimistic" though)
            pareto = (
                self.pareto_front()
            )  # we can't use choice directly, because numpy does not like iterables
            pessimistic = pareto[self._rng.choice(len(pareto))].spawn_child()
        ref = self.parametrization
        if self.crossover and self._num_ask % 2 == 1 and len(self.archive) > 2:
            data = mutator.crossover(
                pessimistic.get_standardized_data(reference=ref),
                mutator.get_roulette(self.archive, num=self.roulette_size),
                rotation=self.rotation,
                crossover_type=self.crossover_type,
            )
            return pessimistic.set_standardized_data(data, reference=ref)
        # mutating

        mutation = self.mutation
        if (
            self._annealing_base is not None
        ):  # We assume that we should start from this one (even if not the best).
            assert self.annealing != "none"
            pessimistic.set_standardized_data(self._annealing_base, reference=ref)
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
            elif (
                "lognormal" in mutation
            ):  # in ["xlognormal", ""lognormal", "smalllognormal", "biglognormal", "hugelognormal"]:
                mutation_rate = self._global_mr
                assert mutation_rate > 0.0
                individual_mutation_rate = 1.0 / (
                    1.0 + (((1.0 - mutation_rate) / mutation_rate) * np.exp(0.22 * np.random.randn()))
                )
                # print(f"pd={pessimistic_data}_imr={individual_mutation_rate}, d={self.dimension}, arity={self.arity_for_discrete_mutation}")
                data = mutator.portfolio_discrete_mutation(
                    pessimistic_data,
                    intensity=individual_mutation_rate * self.dimension,
                    arity=self.arity_for_discrete_mutation,
                )
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
                    pessimistic_data,
                    intensity=intensity,
                    arity=self.arity_for_discrete_mutation,
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
                    pessimistic_data,
                    intensity=intensity,
                    arity=self.arity_for_discrete_mutation,
                )
            elif mutation == "lengler2":
                alpha = 3.0
                intensity = int(max(1, self.dimension * (alpha * np.log(self.num_ask) / self.num_ask)))
                data = mutator.portfolio_discrete_mutation(
                    pessimistic_data,
                    intensity=intensity,
                    arity=self.arity_for_discrete_mutation,
                )
            elif mutation == "lengler3":
                alpha = 9.0
                intensity = int(max(1, self.dimension * (alpha * np.log(self.num_ask) / self.num_ask)))
                data = mutator.portfolio_discrete_mutation(
                    pessimistic_data,
                    intensity=intensity,
                    arity=self.arity_for_discrete_mutation,
                )
            elif mutation == "lenglerfourth":
                alpha = 0.4
                intensity = int(max(1, self.dimension * (alpha * np.log(self.num_ask) / self.num_ask)))
                data = mutator.portfolio_discrete_mutation(
                    pessimistic_data,
                    intensity=intensity,
                    arity=self.arity_for_discrete_mutation,
                )
            elif mutation == "lenglerhalf":
                alpha = 0.8
                intensity = int(max(1, self.dimension * (alpha * np.log(self.num_ask) / self.num_ask)))
                data = mutator.portfolio_discrete_mutation(
                    pessimistic_data,
                    intensity=intensity,
                    arity=self.arity_for_discrete_mutation,
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
                    pessimistic_data,
                    intensity=intensity,
                    arity=self.arity_for_discrete_mutation,
                )
            else:
                func: tp.Any = {  # type: ignore
                    "discrete": mutator.discrete_mutation,
                    "fastga": mutator.doerr_discrete_mutation,
                    "doublefastga": mutator.doubledoerr_discrete_mutation,
                    "rls": mutator.rls_mutation,
                    "portfolio": mutator.portfolio_discrete_mutation,
                }[mutation]
                data = func(pessimistic_data, arity=self.arity_for_discrete_mutation)
            if self.sparse > 0:
                data = np.asarray(data)
                zeroing = self._rng.randint(data.size + 1, size=data.size).reshape(
                    data.shape
                ) < 1 + self._rng.randint(self.sparse)
                data[zeroing] = 0.0
            candidate = pessimistic.set_standardized_data(data, reference=ref)
            if mutation == "coordinatewise_adaptive":
                candidate._meta["modified_variables"] = (self._modified_variables,)
            if (
                "lognormal" in mutation
            ):  # in ["xlognormal", "lognormal", "smalllognormal", "biglognormal", "hugelognormal"]:
                candidate._meta["individual_mutation_rate"] = individual_mutation_rate
            return candidate

    def _internal_tell(self, x: tp.ArrayLike, loss: tp.FloatLoss) -> None:
        if self.annealing != "none":
            assert isinstance(self.budget, int)
            delta = self._previous_best_loss - loss
            if loss > self._max_loss:
                self._max_loss = loss
            if delta >= 0:
                self._annealing_base = x
            elif self.num_ask < self.budget:
                amplitude = max(1.0, self._max_loss - self._previous_best_loss)
                annealing_dict = {
                    "Exp0.9": 0.33 * amplitude * (0.9**self.num_ask),
                    "Exp0.99": 0.33 * amplitude * (0.99**self.num_ask),
                    "Exp0.9Auto": 0.33 * amplitude * ((0.001 ** (1.0 / self.budget)) ** self.num_ask),
                    "Lin100.0": 100.0 * amplitude * (1 - self.num_ask / (self.budget + 1)),
                    "Lin1.0": 1.0 * amplitude * (1 - self.num_ask / (self.budget + 1)),
                    "LinAuto": 10.0 * amplitude * (1 - self.num_ask / (self.budget + 1)),
                }
                T = annealing_dict[self.annealing]
                if T > 0.0:
                    proba = np.exp(delta / T)
                    if self._rng.rand() < proba:
                        self._annealing_base = x
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
        elif self.mutation == "adaptive":
            factor = 1.2 if loss <= self._previous_best_loss else 0.731  # 0.731 = 1.2**(-np.exp(1)-1)
            self._adaptive_mr = min(1.0, factor * self._adaptive_mr)
        elif self.mutation == "coordinatewise_adaptive":
            factor = 1.2 if loss < self._previous_best_loss else 0.731  # 0.731 = 1.2**(-np.exp(1)-1)
            # inds = self._modified_variables
            inds = self.inds
            self._velocity[inds] = np.clip(
                self._velocity[inds] * factor, 1.0, self.arity_for_discrete_mutation / 4.0
            )
        elif (
            "lognormal" in self.mutation
        ):  # in ["xlognormal", "lognormal", "smalllognormal", "biglognormal", "hugelognormal"]:
            # TODO: care about tell not ask, which invalidates the line above.
            self._memory_index = (self._memory_index + 1) % self._memory_size
            if loss < self._best_recent_loss:
                self._best_recent_loss = loss
                self._best_recent_mr = self.imr

            if self._memory_index == 0:
                self._global_mr = self._best_recent_mr
                self._best_recent_loss = float("inf")

        self._previous_best_loss = self.current_bests["pessimistic"].mean  # could be the current one

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        """Called whenever calling :code:`tell` on a candidate that was "asked"."""
        data = candidate.get_standardized_data(reference=self.parametrization)
        if self.mutation == "coordinatewise_adaptive":
            self.inds = (
                candidate._meta["modified_variables"]
                if "modified_variables" in candidate._meta
                else np.array([True] * len(data))
            )
        if (
            "lognormal" in self.mutation
        ):  # in ["xlognormal", "lognormal", "smalllognormal", "biglognormal", "hugelognormal"]:
            self.imr = (
                candidate._meta["individual_mutation_rate"]
                if "individual_mutation_rate" in candidate._meta
                else 0.2
            )
        self._internal_tell(data, loss)


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
        - `"rls"`: Randomized Local Search (randomly mutate one and only one variable).
        - `"portfolio"`: Random number of mutated bits (called niform mixing in
          Dang & Lehre "Self-adaptation of Mutation Rates in Non-elitist Population", 2016)
        - `"lengler"`: specific mutation rate chosen as a function of the dimension and iteration index.
        - `"lengler{2|3|half|fourth}"`: variant of Lengler
    crossover: bool
        whether to add a genetic crossover step every other iteration.
    use_pareto: bool
        whether to restart from a random pareto element in multiobjective mode, instead of the last one added
    sparse: bool
        whether we have random mutations setting variables to 0.
    smoother: bool
        whether we suggest smooth mutations.

    Notes
    -----
    After many papers advocated the mutation rate 1/d in the discrete (1+1) for the discrete case,
    `it was proposed <https://arxiv.org/abs/1606.05551>`_ to use a randomly
    drawn mutation rate. `Fast genetic algorithms <https://arxiv.org/abs/1703.03334>`_ are based on a similar idea
    These two simple methods perform quite well on a wide range of problems.

    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        *,
        noise_handling: tp.Optional[tp.Union[str, tp.Tuple[str, float]]] = None,
        tabu_length: int = 0,
        mutation: str = "gaussian",
        crossover: bool = False,
        rotation: bool = False,
        annealing: str = "none",
        use_pareto: bool = False,
        sparse: bool = False,
        smoother: bool = False,
        roulette_size: int = 2,
        antismooth: int = 55,
        crossover_type: str = "none",
    ) -> None:
        super().__init__(_OnePlusOne, locals())


OnePlusOne = ParametrizedOnePlusOne().set_name("OnePlusOne", register=True)
OnePlusLambda = ParametrizedOnePlusOne().set_name(
    "OnePlusLambda", register=True
)  # Same as one-plus-one as lambda is set to num_workers
# SA = ParametrizedOnePlusOne(annealing="Exp0.9").set_name("SA", register=True)
NoisyOnePlusOne = ParametrizedOnePlusOne(noise_handling="random").set_name("NoisyOnePlusOne", register=True)
DiscreteOnePlusOne = ParametrizedOnePlusOne(mutation="discrete").set_name("DiscreteOnePlusOne", register=True)
SADiscreteLenglerOnePlusOneExp09 = ParametrizedOnePlusOne(
    tabu_length=1000, mutation="lengler", annealing="Exp0.9"
).set_name("SADiscreteLenglerOnePlusOneExp09", register=True)
SADiscreteLenglerOnePlusOneExp099 = ParametrizedOnePlusOne(
    tabu_length=1000, mutation="lengler", annealing="Exp0.99"
).set_name("SADiscreteLenglerOnePlusOneExp099", register=True)
SADiscreteLenglerOnePlusOneExp09Auto = ParametrizedOnePlusOne(
    tabu_length=1000, mutation="lengler", annealing="Exp0.9Auto"
).set_name("SADiscreteLenglerOnePlusOneExp09Auto", register=True)
SADiscreteLenglerOnePlusOneLinAuto = ParametrizedOnePlusOne(
    tabu_length=1000, mutation="lengler", annealing="LinAuto"
).set_name("SADiscreteLenglerOnePlusOneLinAuto", register=True)
SADiscreteLenglerOnePlusOneLin1 = ParametrizedOnePlusOne(
    tabu_length=1000, mutation="lengler", annealing="Lin1.0"
).set_name("SADiscreteLenglerOnePlusOneLin1", register=True)
SADiscreteLenglerOnePlusOneLin100 = ParametrizedOnePlusOne(
    tabu_length=1000, mutation="lengler", annealing="Lin100.0"
).set_name("SADiscreteLenglerOnePlusOneLin100", register=True)
SADiscreteOnePlusOneExp099 = ParametrizedOnePlusOne(
    tabu_length=1000, mutation="discrete", annealing="Exp0.99"
).set_name("SADiscreteOnePlusOneExp099", register=True)
SADiscreteOnePlusOneLin100 = ParametrizedOnePlusOne(
    tabu_length=1000, mutation="discrete", annealing="Lin100.0"
).set_name("SADiscreteOnePlusOneLin100", register=True)
SADiscreteOnePlusOneExp09 = ParametrizedOnePlusOne(
    tabu_length=1000, mutation="discrete", annealing="Exp0.9"
).set_name("SADiscreteOnePlusOneExp09", register=True)
DiscreteOnePlusOneT = ParametrizedOnePlusOne(tabu_length=10000, mutation="discrete").set_name(
    "DiscreteOnePlusOneT", register=True
)
PortfolioDiscreteOnePlusOne = ParametrizedOnePlusOne(mutation="portfolio").set_name(
    "PortfolioDiscreteOnePlusOne", register=True
)
PortfolioDiscreteOnePlusOneT = ParametrizedOnePlusOne(tabu_length=10000, mutation="portfolio").set_name(
    "PortfolioDiscreteOnePlusOneT", register=True
)
DiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(mutation="lengler").set_name(
    "DiscreteLenglerOnePlusOne", register=True
)
DiscreteLengler2OnePlusOne = ParametrizedOnePlusOne(mutation="lengler2").set_name(
    "DiscreteLengler2OnePlusOne", register=True
)
DiscreteLengler3OnePlusOne = ParametrizedOnePlusOne(mutation="lengler3").set_name(
    "DiscreteLengler3OnePlusOne", register=True
)
DiscreteLenglerHalfOnePlusOne = ParametrizedOnePlusOne(mutation="lenglerhalf").set_name(
    "DiscreteLenglerHalfOnePlusOne", register=True
)
DiscreteLenglerFourthOnePlusOne = ParametrizedOnePlusOne(mutation="lenglerfourth").set_name(
    "DiscreteLenglerFourthOnePlusOne", register=True
)
DiscreteLenglerOnePlusOneT = ParametrizedOnePlusOne(tabu_length=10000, mutation="lengler").set_name(
    "DiscreteLenglerOnePlusOneT", register=True
)
AdaptiveDiscreteOnePlusOne = ParametrizedOnePlusOne(mutation="adaptive").set_name(
    "AdaptiveDiscreteOnePlusOne", register=True
)
LognormalDiscreteOnePlusOne = ParametrizedOnePlusOne(mutation="lognormal").set_name(
    "LognormalDiscreteOnePlusOne", register=True
)
XLognormalDiscreteOnePlusOne = ParametrizedOnePlusOne(mutation="xlognormal").set_name(
    "XLognormalDiscreteOnePlusOne", register=True
)
XSmallLognormalDiscreteOnePlusOne = ParametrizedOnePlusOne(mutation="xsmalllognormal").set_name(
    "XSmallLognormalDiscreteOnePlusOne", register=True
)
BigLognormalDiscreteOnePlusOne = ParametrizedOnePlusOne(mutation="biglognormal").set_name(
    "BigLognormalDiscreteOnePlusOne", register=True
)
SmallLognormalDiscreteOnePlusOne = ParametrizedOnePlusOne(mutation="smalllognormal").set_name(
    "SmallLognormalDiscreteOnePlusOne", register=True
)
HugeLognormalDiscreteOnePlusOne = ParametrizedOnePlusOne(mutation="hugelognormal").set_name(
    "HugeLognormalDiscreteOnePlusOne", register=True
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
OLNDiscreteOnePlusOne = ParametrizedOnePlusOne(noise_handling="optimistic", mutation="lognormal").set_name(
    "OLNDiscreteOnePlusOne", register=True
)
NoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(
    noise_handling=("random", 1.0), mutation="discrete"
).set_name("NoisyDiscreteOnePlusOne", register=True)
DoubleFastGADiscreteOnePlusOne = ParametrizedOnePlusOne(mutation="doublefastga").set_name(
    "DoubleFastGADiscreteOnePlusOne", register=True
)
RLSOnePlusOne = ParametrizedOnePlusOne(mutation="rls").set_name("RLSOnePlusOne", register=True)
SparseDoubleFastGADiscreteOnePlusOne = ParametrizedOnePlusOne(mutation="doublefastga", sparse=True).set_name(
    "SparseDoubleFastGADiscreteOnePlusOne", register=True
)
RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(
    crossover=True, mutation="portfolio", noise_handling="optimistic"
).set_name("RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne", register=True)
RecombiningPortfolioDiscreteOnePlusOne = ParametrizedOnePlusOne(
    crossover=True,
    mutation="portfolio",
).set_name("RecombiningPortfolioDiscreteOnePlusOne", register=True)


# pylint: too-many-arguments,too-many-instance-attributes


class _CMA(base.Optimizer):
    _CACHE_KEY = "#CMA#datacache"

    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        config: tp.Optional["ParametrizedCMA"] = None,
        algorithm: str = "quad",
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.algorithm = algorithm
        self._config = ParametrizedCMA() if config is None else config
        pop = self._config.popsize
        self._popsize = (
            max(num_workers, 4 + int(self._config.popsize_factor * np.log(self.dimension)))
            if pop is None
            else max(pop, num_workers)
        )
        if self._config.elitist:
            self._popsize = max(self._popsize, self.num_workers + 1)
        # internal attributes
        self._to_be_asked: tp.Deque[np.ndarray] = deque()
        self._to_be_told: tp.List[p.Parameter] = []
        self._num_spawners = self._popsize // 2  # experimental, for visualization
        self._parents = [self.parametrization]
        # delay initialization to ease implementation of variants
        self._es: tp.Any = None

    @property
    def es(self) -> tp.Any:  # typing not possible since cmaes not imported :(
        scale_multiplier = 1.0
        if self.dimension == 1:
            self._config.fcmaes = True
        if p.helpers.Normalizer(self.parametrization).fully_bounded:
            scale_multiplier = 0.3 if self.dimension < 18 else 0.15
        if self._es is None or (not self._config.fcmaes and self._es.stop()):
            if not self._config.fcmaes:
                import cma  # import inline in order to avoid matplotlib initialization warning

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
                    x0=(
                        self.parametrization.sample().get_standardized_data(reference=self.parametrization)
                        if self._config.random_init
                        else np.zeros(self.dimension, dtype=np.float64)
                    ),
                    sigma0=self._config.scale * scale_multiplier,
                    inopts=inopts,
                )
            else:
                try:
                    from fcmaes import cmaes
                except ImportError as e:
                    raise ImportError(
                        "Please install fcmaes (pip install fcmaes) to use FCMA optimizers"
                    ) from e
                self._es = cmaes.Cmaes(
                    x0=np.zeros(self.dimension, dtype=np.float64),
                    input_sigma=self._config.scale * scale_multiplier,
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
        if self._CACHE_KEY not in candidate._meta:
            # since we try several times to tell to es, to avoid duplicated work let's keep
            # the data in a cache. This can be useful for other CMA as well
            candidate._meta[self._CACHE_KEY] = candidate.get_standardized_data(reference=self.parametrization)
        self._to_be_told.append(candidate)
        if len(self._to_be_told) >= self.es.popsize:
            listx = [c._meta[self._CACHE_KEY] for c in self._to_be_told]
            listy = [c.loss for c in self._to_be_told]
            args = (listy, listx) if self._config.fcmaes else (listx, listy)
            try:
                self.es.tell(*args)
            except (RuntimeError, AssertionError):
                pass
            else:
                self._parents = sorted(self._to_be_told, key=base._loss)[: self._num_spawners]
            self._to_be_told = []

    def _internal_provide_recommendation(self) -> np.ndarray:
        pessimistic = self.current_bests["pessimistic"].parameter.get_standardized_data(
            reference=self.parametrization
        )
        d = self.dimension
        n = self.num_ask
        sample_size = int(d * d / 2 + d / 2 + 3)
        if self._config.high_speed and n >= sample_size:
            try:
                data = learn_on_k_best(self.archive, sample_size, self.algorithm)
                return data  # type: ignore
            except MetaModelFailure:  # Failures in the metamodeling can happen.
                pass
        if self._es is None:
            return pessimistic
        cma_best: tp.Optional[np.ndarray] = self.es.best_x if self._config.fcmaes else self.es.result.xbest
        if cma_best is None:
            return pessimistic
        return cma_best


class ParametrizedCMA(base.ConfiguredOptimizer):
    """CMA-ES optimizer,
    This evolution strategy uses Gaussian sampling, iteratively modified
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
    popsize_factor: float = 3.
        factor in the formula for computing the population size
    diagonal: bool
        use the diagonal version of CMA (advised in big dimension)
    high_speed: bool
        use metamodel for recommendation
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
        popsize_factor: float = 3.0,
        diagonal: bool = False,
        zero: bool = False,
        high_speed: bool = False,
        fcmaes: bool = False,
        random_init: bool = False,
        inopts: tp.Optional[tp.Dict[str, tp.Any]] = None,
        algorithm: str = "quad",
    ) -> None:
        super().__init__(_CMA, locals(), as_config=True)
        if zero:
            scale = scale / 1000.0
        if fcmaes:
            if diagonal:
                raise RuntimeError("fcmaes doesn't support diagonal=True, use fcmaes=False")
        self.scale = scale
        self.elitist = elitist
        self.zero = zero
        self.popsize = popsize
        self.popsize_factor = popsize_factor
        self.diagonal = diagonal
        self.fcmaes = fcmaes
        self.high_speed = high_speed
        self.random_init = random_init
        self.inopts = inopts


@registry.register
class ChoiceBase(base.Optimizer):
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
            # try:
            self._optim = self._select_optimizer_cls()(self.parametrization, self.budget, self.num_workers)
            # except:
            #    self._optim = NGOpt39._select_optimizer_cls(self)(self.parametrization, self.budget, self.num_workers)
            self._optim = self._optim if not isinstance(self._optim, NGOptBase) else self._optim.optim
            logger.debug("%s selected %s optimizer.", *(x.name for x in (self, self._optim)))
        return self._optim

    def _select_optimizer_cls(self) -> base.OptCls:
        return CMA

    def _internal_ask_candidate(self) -> p.Parameter:
        return self.optim.ask()

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        self.optim.tell(candidate, loss)

    def recommend(self) -> p.Parameter:
        return self.optim.recommend()

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        self.optim.tell(candidate, loss)

    def _info(self) -> tp.Dict[str, tp.Any]:
        out = {"sub-optim": self.optim.name}
        out.update(self.optim._info())  # this will work for recursive NGOpt calls
        return out

    def enable_pickling(self) -> None:
        self.optim.enable_pickling()


OldCMA = ParametrizedCMA().set_name("OldCMA", register=True)
LargeCMA = ParametrizedCMA(scale=3.0).set_name("LargeCMA", register=True)
LargeDiagCMA = ParametrizedCMA(scale=3.0, diagonal=True).set_name("LargeDiagCMA", register=True)
TinyCMA = ParametrizedCMA(scale=0.33).set_name("TinyCMA", register=True)
CMAbounded = ParametrizedCMA(
    scale=1.5884, popsize_factor=1, elitist=True, diagonal=True, fcmaes=False
).set_name("CMAbounded", register=True)
CMAsmall = ParametrizedCMA(
    scale=0.3607, popsize_factor=3, elitist=False, diagonal=False, fcmaes=False
).set_name("CMAsmall", register=True)
CMAstd = ParametrizedCMA(
    scale=0.4699, popsize_factor=3, elitist=False, diagonal=False, fcmaes=False
).set_name("CMAstd", register=True)
CMApara = ParametrizedCMA(scale=0.8905, popsize_factor=8, elitist=True, diagonal=True, fcmaes=False).set_name(
    "CMApara", register=True
)
CMAtuning = ParametrizedCMA(
    scale=0.4847, popsize_factor=1, elitist=True, diagonal=False, fcmaes=False
).set_name("CMAtuning", register=True)


@registry.register
class MetaCMA(ChoiceBase):  # Adds Risto's CMA to CMA.
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self) -> base.OptCls:
        if (
            self.budget is not None
            and self.fully_continuous
            and not self.has_noise
            and self.num_objectives < 2
        ):
            if self.dimension == 1:
                return OnePlusOne
            if p.helpers.Normalizer(self.parametrization).fully_bounded:
                return CMAbounded
            if self.budget < 50:
                if self.dimension <= 15:
                    return CMAtuning
                return CMAsmall
            if self.num_workers > 20:
                return CMApara
            return CMAstd
        else:
            return OldCMA


DiagonalCMA = ParametrizedCMA(diagonal=True).set_name("DiagonalCMA", register=True)
SDiagonalCMA = ParametrizedCMA(diagonal=True, zero=True).set_name("SDiagonalCMA", register=True)
FCMA = ParametrizedCMA(fcmaes=True).set_name("FCMA", register=True)


@registry.register
class CMA(MetaCMA):
    pass


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
        config: tp.Optional["ConfPSO"] = None,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._config = ConfPSO() if config is None else config
        # if budget is not None and budget < 60:
        #    warnings.warn("PSO is inefficient with budget < 60", errors.InefficientSettingsWarning)
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
        self.previous_candidate: tp.Optional[tp.Any] = None
        self.previous_speed: tp.Optional[tp.Any] = None

    def _internal_ask_candidate(self) -> p.Parameter:
        # population is increased only if queue is empty (otherwise tell_not_asked does not work well at the beginning)
        if len(self.population) < self.llambda:
            r = (
                np.exp(-5.0 * self._rng.rand())
                if self._config.so
                else (self._rng.rand() if self._config.sqo else 1.0)
            )
            candidate = self.parametrization.sample()
            if self._config.qo:
                if self.previous_candidate is not None:
                    data = self.previous_candidate.get_standardized_data(reference=self.parametrization)
                    candidate.set_standardized_data(-r * data, reference=self.parametrization)
                    self.previous_candidate = None
                else:
                    self.previous_candidate = candidate
            self.population[candidate.uid] = candidate
            dim = self.parametrization.dimension
            candidate.heritage["speed"] = (
                self._rng.normal(size=dim) if self._eps is None else self._rng.uniform(-1, 1, dim)
            )
            if self._config.sqo:
                assert self._config.qo, "SQO only when QO!"
                if self.previous_speed is not None:
                    candidate.heritage["speed"] = -r * self.previous_speed
                    self.previous_speed = None
                else:
                    self.previous_speed = candidate.heritage["speed"]

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


class ConfPSO(base.ConfiguredOptimizer):
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
    qo: bool
        whether we use quasi-opposite initialization
    sqo: bool
        whether we use quasi-opposite initialization for speed
    so: bool
        whether we use the special quasi-opposite initialization for speed

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
        qo: bool = False,
        sqo: bool = False,
        so: bool = False,
    ) -> None:
        super().__init__(_PSO, locals(), as_config=True)
        assert transform in ["arctan", "gaussian", "identity"]
        self.transform = transform
        self.popsize = popsize
        self.omega = omega
        self.phip = phip
        self.phig = phig
        self.qo = qo
        self.sqo = sqo
        self.so = so


ConfiguredPSO = ConfPSO  # backward compatibility (to be removed)
RealSpacePSO = ConfPSO().set_name("RealSpacePSO", register=True)
PSO = ConfPSO(transform="arctan").set_name("PSO", register=True)
QOPSO = ConfPSO(transform="arctan", qo=True).set_name("QOPSO", register=True)
QORealSpacePSO = ConfPSO(qo=True).set_name("QORealSpacePSO", register=True)
SQOPSO = ConfPSO(transform="arctan", qo=True, sqo=True).set_name("SQOPSO", register=True)
SOPSO = ConfPSO(transform="arctan", qo=True, sqo=True, so=True).set_name("SOPSO", register=True)
SQORealSpacePSO = ConfPSO(qo=True, sqo=True).set_name("SQORealSpacePSO", register=True)


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
        self.delta: tp.Any = float("nan")
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
        base_optimizer: base.OptCls = MetaCMA,
        scale: tp.Optional[float] = None,
        shift: tp.Optional[float] = None,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._optimizer = base_optimizer(self.parametrization, budget=budget, num_workers=num_workers)
        self.no_parallelization = self._optimizer.no_parallelization
        self._subcandidates: tp.Dict[str, p.Parameter] = {}
        if scale is None:
            assert self.budget is not None, "Either scale or budget must be known in _Rescaled."
            scale = math.sqrt(math.log(self.budget) / self.dimension)
        self.scale = scale
        self.shift = shift
        assert self.scale != 0.0, "scale should be non-zero in Rescaler."

    def rescale_candidate(self, candidate: p.Parameter, inverse: bool = False) -> p.Parameter:
        data = candidate.get_standardized_data(reference=self.parametrization)
        if self.shift is not None:
            data = data + self.shift * np.random.randn(self.dimension)
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

    def enable_pickling(self) -> None:
        self._optimizer.enable_pickling()


class Rescaled(base.ConfiguredOptimizer):
    """Configured optimizer for creating rescaled optimization algorithms.

    By default, scales to sqrt(log(budget)/n_dimensions).

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
        base_optimizer: base.OptCls = MetaCMA,
        scale: tp.Optional[float] = None,
        shift: tp.Optional[float] = None,
    ) -> None:
        super().__init__(_Rescaled, locals())


RescaledCMA = Rescaled().set_name("RescaledCMA", register=True)
TinyLhsDE = Rescaled(base_optimizer=LhsDE, scale=1e-3).set_name("TinyLhsDE", register=True)
TinyQODE = Rescaled(base_optimizer=QODE, scale=1e-3).set_name("TinyQODE", register=True)
TinySQP = Rescaled(base_optimizer=SQP, scale=1e-3).set_name("TinySQP", register=True)
MicroSQP = Rescaled(base_optimizer=SQP, scale=1e-6).set_name("MicroSQP", register=True)
TinySQP.no_parallelization = True
MicroSQP.no_parallelization = True
TinySPSA = Rescaled(base_optimizer=SPSA, scale=1e-3).set_name("TinySPSA", register=True)
MicroSPSA = Rescaled(base_optimizer=SPSA, scale=1e-6).set_name("MicroSPSA", register=True)
TinySPSA.no_parallelization = True
MicroSPSA.no_parallelization = True
VastLengler = Rescaled(base_optimizer=DiscreteLenglerOnePlusOne, scale=1000).set_name(
    "VastLengler", register=True
)
VastDE = Rescaled(base_optimizer=DE, scale=1000).set_name("VastDE", register=True)


class SplitOptimizer(base.Optimizer):
    """Combines optimizers, each of them working on their own variables. (use ConfSplitOptimizer)"""

    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        config: tp.Optional["ConfSplitOptimizer"] = None,
    ) -> None:
        self._config = ConfSplitOptimizer() if config is None else config
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._subcandidates: tp.Dict[str, tp.List[p.Parameter]] = {}
        subparams: tp.List[p.Parameter] = []

        num_vars = self._config.num_vars
        num_optims = self._config.num_optims
        max_num_vars = self._config.max_num_vars
        if max_num_vars is not None:
            assert num_vars is None, "num_vars and max_num_vars should not be set at the same time"
            num_vars = [max_num_vars] * (self.dimension // max_num_vars)
            if self.dimension > sum(num_vars):
                num_vars += [self.dimension - sum(num_vars)]
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
            num_optims = int(min(num_optims, self.dimension))
            num_vars = num_vars if num_vars else []
            for i in range(num_optims):
                if len(num_vars) < i + 1:
                    num_vars += [(self.dimension // num_optims) + (self.dimension % num_optims > i)]
                assert num_vars[i] >= 1, "At least one variable per optimizer."
                subparams += [p.Array(shape=(num_vars[i],))]
        if self._config.non_deterministic_descriptor:
            for param in subparams:
                param.function.deterministic = False
        # synchronize random state and create optimizers
        self.optims: tp.List[base.Optimizer] = []
        mono, multi = self._config.monovariate_optimizer, self._config.multivariate_optimizer
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
            if self._config.progressive:
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

    def _info(self) -> tp.Dict[str, tp.Any]:
        key = "sub-optim"
        optims = [x.name if key not in x._info() else x._info()[key] for x in self.optims]
        return {key: ",".join(optims)}


class ConfSplitOptimizer(base.ConfiguredOptimizer):
    """Combines optimizers, each of them working on their own variables.

    Parameters
    ----------
    num_optims: int (or float("inf"))
        number of optimizers to create (if not provided through :code:`num_vars: or
        :code:`max_num_vars`)
    num_vars: int or None
        number of variable per optimizer (should not be used if :code:`max_num_vars` or
        :code:`num_optims` is set)
    max_num_vars: int or None
        maximum number of variables per optimizer. Should not be defined if :code:`num_vars` or
        :code:`num_optims` is defined since they will be chosen automatically.
    progressive: optional bool
        whether we progressively add optimizers.
    non_deterministic_descriptor: bool
        subparts parametrization descriptor is set to noisy function.
        This can have an impact for optimizer selection for competence maps.

    Example
    -------
    for 5 optimizers, each of them working on 2 variables, one can use:

    opt = ConfSplitOptimizer(num_vars=[2, 2, 2, 2, 2])(parametrization=10, num_workers=3)
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

    # pylint: disable=unused-argument
    def __init__(
        self,
        *,
        num_optims: tp.Optional[float] = None,
        num_vars: tp.Optional[tp.List[int]] = None,
        max_num_vars: tp.Optional[int] = None,
        multivariate_optimizer: base.OptCls = MetaCMA,
        monovariate_optimizer: base.OptCls = oneshot.RandomSearch,
        progressive: bool = False,
        non_deterministic_descriptor: bool = True,
    ) -> None:
        self.num_optims = num_optims
        self.num_vars = num_vars
        self.max_num_vars = max_num_vars
        self.multivariate_optimizer = multivariate_optimizer
        self.monovariate_optimizer = monovariate_optimizer
        self.progressive = progressive
        self.non_deterministic_descriptor = non_deterministic_descriptor
        if sum(x is not None for x in [num_optims, num_vars, max_num_vars]) > 1:
            raise ValueError("At most, only one of num_optims, num_vars and max_num_vars can be set")
        super().__init__(SplitOptimizer, locals(), as_config=True)


class NoisySplit(base.ConfiguredOptimizer):
    """Non-progressive noisy split of variables based on 1+1

    Parameters
    ----------
    num_optims: optional int
        number of optimizers (one per variable if float("inf"))
    discrete: bool
        uses OptimisticDiscreteOnePlusOne if True, else NoisyOnePlusOne
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        *,
        num_optims: tp.Optional[float] = None,
        discrete: bool = False,
    ) -> None:
        kwargs = locals()
        opt = OptimisticDiscreteOnePlusOne if discrete else OptimisticNoisyOnePlusOne
        mono_opt: base.OptCls = NoisyBandit if discrete else opt
        ConfOpt = ConfSplitOptimizer(
            progressive=False,
            num_optims=num_optims,
            multivariate_optimizer=opt,
            monovariate_optimizer=mono_opt,
        )
        super().__init__(ConfOpt, kwargs)


class ConfPortfolio(base.ConfiguredOptimizer):
    """Alternates :code:`ask()` on several optimizers

    Parameters
    ----------
    optimizers: list of Optimizer, optimizer name, Optimizer class or ConfiguredOptimizer
        the list of optimizers to use.
    warmup_ratio: optional float
        ratio of the budget used before choosing to focus on one optimizer

    Notes
    -----
    - if providing an initialized  optimizer, the parametrization of the optimizer
      must be the exact same instance as the one of the Portfolio.
    - this API is temporary and will be renamed very soon
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        *,
        optimizers: tp.Sequence[tp.Union[base.Optimizer, base.OptCls, str]] = (),
        warmup_ratio: tp.Optional[float] = None,
        no_crossing: bool = False,
    ) -> None:
        self.optimizers = optimizers
        self.warmup_ratio = warmup_ratio
        super().__init__(Portfolio, locals(), as_config=True)
        self.no_crossing = no_crossing


@registry.register
class Portfolio(base.Optimizer):
    """Passive portfolio of CMA, 2-pt DE and Scr-Hammersley."""

    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        config: tp.Optional["ConfPortfolio"] = None,
    ) -> None:
        self.no_crossing = False
        distribute_workers = config is not None and config.warmup_ratio == 1.0
        self._config = ConfPortfolio() if config is None else config
        cfg = self._config
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        optimizers = list(cfg.optimizers)
        if not optimizers:  # default
            optimizers = []
            if budget is None or budget >= 12 * num_workers:
                optimizers = [MetaCMA, "TwoPointsDE"]
            if budget is not None:  # needs a budget
                optimizers.append("ScrHammersleySearch")
        num = len(optimizers)
        # print(f"Optimizers = {optimizers}")
        self.optims: tp.List[base.Optimizer] = []
        num_non_para = 0
        for opt in optimizers:
            Optim_tmp: base.OptCls = registry[opt] if isinstance(opt, str) else opt  # type: ignore
            if Optim_tmp.no_parallelization:
                num_non_para += 1
        if num_workers <= num:
            num_non_para = 0  # no problem with non-parallelizable if num_workers < num
        num_para = num - num_non_para
        rebudget = (
            int(np.ceil((budget * config.warmup_ratio)))
            if config is not None and config.warmup_ratio is not None and budget is not None
            else budget
        )
        sub_budget = None if budget is None else rebudget // num_para + (rebudget % num_para > 0)  # type: ignore
        if (
            budget is not None
            and sub_budget is not None
            and config is not None
            and config.warmup_ratio is not None
            and config.warmup_ratio < 1.0
        ):
            sub_budget += 1 + int(
                budget * (1 - config.warmup_ratio)
            )  # We need additional budget for the selected optimizer
        needed_workers = num_workers - num_non_para
        sub_workers = max(
            1, needed_workers // (num - num_non_para) + (needed_workers % (num - num_non_para) > 0)
        )
        if distribute_workers:
            sub_workers = num_workers // num + (num_workers % num > 0)
        # print(
        #     f"{num_workers} workers, {num} algorithms, budget {budget}, {num} optimizers, {sub_workers} sub_workers, sub_budget={sub_budget}, distrib: {distribute_workers}"
        # )
        assert num * sub_workers >= num_workers
        turns = []
        self.str_info = ""
        for index, opt in enumerate(optimizers):
            if isinstance(opt, base.Optimizer):
                self.str_info += f"XX{opt}"
                if opt.parametrization is not self.parametrization:
                    raise errors.NevergradValueError(
                        "Initialized optimizers are only accepted if "
                        "the parametrization object is strictly the same"
                    )
                self.str_info += f"append{opt}"
                turns += [index] * (1 if opt.no_parallelization else sub_workers)
                if not opt.no_parallelization:
                    assert opt.num_workers == sub_workers
                self.optims.append(opt)
                continue
            Optim: base.OptCls = registry[opt] if isinstance(opt, str) else opt
            # assert sub_workers == 1 or not Optim.no_parallelization
            self.optims.append(
                Optim(
                    self.parametrization,  # share parametrization and its rng
                    budget=sub_budget,
                    num_workers=1 if Optim.no_parallelization else sub_workers,
                )
            )
            assert sub_workers >= 1
            turns += [index] * (1 if Optim.no_parallelization else sub_workers)
            self.str_info += f"Added {opt} with budget {sub_budget} with {1 if Optim.no_parallelization else sub_workers} workers"  # DEBUG INFO
            # print( f"Added {opt} with budget {sub_budget} with {1 if Optim.no_parallelization else sub_workers} workers")
        self.turns = turns
        # print("end additions...", self.turns, optimizers)
        assert max(self.turns) < len(self.optims)
        # current optimizer choice
        self._current = -1
        self._warmup_budget: tp.Optional[int] = None
        if cfg.warmup_ratio is not None and budget is None:
            raise ValueError("warmup_ratio is only available if a budget is provided")
        if not any(x is None for x in (cfg.warmup_ratio, budget)):
            self._warmup_budget = int(cfg.warmup_ratio * budget)  # type: ignore
        self.num_times = [0] * num

    def _internal_ask_candidate(self) -> p.Parameter:
        # optimizer selection if budget is over
        if self._warmup_budget is not None:
            if len(self.optims) > 1 and self._warmup_budget < self.num_tell:
                ind = self.current_bests["pessimistic"].parameter._meta.get("optim_index", -1)
                if ind >= 0 and ind < len(self.optims):  # not a tell not asked
                    if self.num_workers == 1 or self.optims[ind].num_workers > 1:
                        self.optims = [self.optims[ind]]  # throw away everything else
        num = len(self.optims)
        if num == 1:
            optim_index = 0
        else:
            # for _ in range(2 * len(self.turns)):
            self._current += 1
            # optim_index = self._current % len(self.optims)
            optim_index = self.turns[self._current % len(self.turns)]
            assert optim_index < len(
                self.optims
            ), f"{optim_index}, {self.turns}, {len(self.optims)} {self.num_times} {self.str_info} {self.optims}"
            opt = self.optims[optim_index]
            # break
            # if opt.num_workers > opt.num_ask - (opt.num_tell):  # - opt.num_tell_not_asked):
            #     # if opt.num_workers > opt.num_ask - (opt.num_tell - opt.num_tell_not_asked):
            #     break  # if there are workers left, use this optimizer
            # # print(optim_index, " not available", opt)  # DEBUG INFO
            # # print( f"{opt} ({optim_index}) not available, because {opt.num_workers} not > {opt.num_ask} - ({opt.num_tell} - {opt.num_tell_not_asked})")  # DEBUG INFO
            # if k > len(self.turns):
            #     if not opt.no_parallelization:
            #         break  # if no worker is available, try the first parallelizable optimizer
        if optim_index is None:
            raise RuntimeError("Something went wrong in optimizer selection")
        opt = self.optims[optim_index]
        self.num_times[optim_index] += 1
        # print( optim_index, " is chosen:", opt, self.num_times[optim_index], self.turns, self.optims, self.num_times, self.num_workers,)  # DEBUG INFO
        if optim_index > 1 and not opt.num_ask and not opt._suggestions and not opt.num_tell:
            # most algorithms start at 0, lets avoid that for all but the first if they have no information
            opt._suggestions.append(self.parametrization.sample())
            # (hacky suggestion to avoid calling args and kwargs)
        candidate = opt.ask()
        candidate._meta["optim_index"] = optim_index
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        # Telling all optimizers is presumably better than just
        # self.optims[optim_index].tell(candidate, value)
        accepted = 0
        if self.no_crossing and len(self.optims) > candidate._meta["optim_index"]:
            self.optims[candidate._meta["optim_index"]].tell(candidate, loss)
            return
        for opt in self.optims:
            try:
                opt.tell(candidate, loss)
                accepted += 1
            except errors.TellNotAskedNotSupportedError:
                pass
        if not accepted:
            raise errors.TellNotAskedNotSupportedError("No sub-optimizer accepted the tell-not-asked")

    def enable_pickling(self) -> None:
        for opt in self.optims:
            opt.enable_pickling()


ParaPortfolio = ConfPortfolio(optimizers=[MetaCMA, TwoPointsDE, PSO, SQP, ScrHammersleySearch]).set_name(
    "ParaPortfolio", register=True
)
ASCMADEthird = ConfPortfolio(optimizers=[MetaCMA, LhsDE], warmup_ratio=0.33).set_name(
    "ASCMADEthird", register=True
)
MultiCMA = ConfPortfolio(
    optimizers=[ParametrizedCMA(random_init=True) for _ in range(3)], warmup_ratio=0.1
).set_name("MultiCMA", register=True)
MultiDS = ConfPortfolio(optimizers=[DSproba for _ in range(3)], warmup_ratio=0.1).set_name(
    "MultiDS", register=True
)
TripleCMA = ConfPortfolio(
    optimizers=[ParametrizedCMA(random_init=True) for _ in range(3)], warmup_ratio=0.33
).set_name("TripleCMA", register=True)
PolyCMA = ConfPortfolio(
    optimizers=[ParametrizedCMA(random_init=True) for _ in range(20)], warmup_ratio=0.33
).set_name("PolyCMA", register=True)
MultiScaleCMA = ConfPortfolio(
    optimizers=[ParametrizedCMA(random_init=True, scale=scale) for scale in [1.0, 1e-3, 1e-6]],
    warmup_ratio=0.33,
).set_name("MultiScaleCMA", register=True)
LPCMA = ParametrizedCMA(popsize_factor=10.0).set_name("LPCMA", register=True)
VLPCMA = ParametrizedCMA(popsize_factor=100.0).set_name("VLPCMA", register=True)


class _MetaModel(base.Optimizer):
    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        *,
        multivariate_optimizer: tp.Optional[base.OptCls] = None,
        frequency_ratio: float = 0.9,
        algorithm: str,  # Quad or NN or SVR
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.frequency_ratio = frequency_ratio
        self.algorithm = algorithm
        elitist = self.dimension < 3
        if multivariate_optimizer is None:
            multivariate_optimizer = ParametrizedCMA(elitist=elitist) if self.dimension > 1 else OnePlusOne
        self._optim = multivariate_optimizer(
            self.parametrization, budget, num_workers
        )  # share parametrization and its rng

    def _internal_ask_candidate(self) -> p.Parameter:
        # We request a bit more points than what is really necessary for our dimensionality (+dimension).
        sample_size = int((self.dimension * (self.dimension - 1)) / 2 + 2 * self.dimension + 1)
        freq = max(13, self.num_workers, self.dimension, int(self.frequency_ratio * sample_size))
        if len(self.archive) >= sample_size and not self._num_ask % freq:
            try:
                data = learn_on_k_best(self.archive, sample_size, self.algorithm)
                candidate = self.parametrization.spawn_child().set_standardized_data(data)
            except (OverflowError, MetaModelFailure):  # The optimum is at infinity. Shit happens.
                candidate = self._optim.ask()
        else:
            candidate = self._optim.ask()
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        self._optim.tell(candidate, loss)

    def _internal_provide_recommendation(self) -> tp.Optional[tp.ArrayLike]:
        return self._optim._internal_provide_recommendation()

    def enable_pickling(self):
        super().enable_pickling()
        self._optim.enable_pickling()


class ParametrizedMetaModel(base.ConfiguredOptimizer):
    """
    Adds a metamodel to an optimizer.
    The optimizer is alway OnePlusOne if dimension is 1.

    Parameters
    ----------
    multivariate_optimizer: base.OptCls or None
        Optimizer to which the metamodel is added
    frequency_ratio: float
        used for deciding the frequency at which we use the metamodel
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        *,
        multivariate_optimizer: tp.Optional[base.OptCls] = None,
        frequency_ratio: float = 0.9,
        algorithm: str = "quad",
    ) -> None:
        super().__init__(_MetaModel, locals())
        assert 0 <= frequency_ratio <= 1.0


MetaModel = ParametrizedMetaModel().set_name("MetaModel", register=True)
NeuralMetaModel = ParametrizedMetaModel(algorithm="neural").set_name("NeuralMetaModel", register=True)
SVMMetaModel = ParametrizedMetaModel(algorithm="svr").set_name("SVMMetaModel", register=True)
RFMetaModel = ParametrizedMetaModel(algorithm="rf").set_name("RFMetaModel", register=True)
MetaModelOnePlusOne = ParametrizedMetaModel(multivariate_optimizer=OnePlusOne).set_name(
    "MetaModelOnePlusOne", register=True
)
MetaModelDSproba = ParametrizedMetaModel(multivariate_optimizer=DSproba).set_name(
    "MetaModelDSproba", register=True
)
RFMetaModelOnePlusOne = ParametrizedMetaModel(multivariate_optimizer=OnePlusOne, algorithm="rf").set_name(
    "RFMetaModelOnePlusOne", register=True
)
MetaModelPSO = ParametrizedMetaModel(multivariate_optimizer=PSO).set_name("MetaModelPSO", register=True)
RFMetaModelPSO = ParametrizedMetaModel(multivariate_optimizer=PSO, algorithm="rf").set_name(
    "RFMetaModelPSO", register=True
)
SVMMetaModelPSO = ParametrizedMetaModel(multivariate_optimizer=PSO, algorithm="svr").set_name(
    "SVMMetaModelPSO", register=True
)

MetaModelDE = ParametrizedMetaModel(multivariate_optimizer=DE).set_name("MetaModelDE", register=True)
MetaModelQODE = ParametrizedMetaModel(multivariate_optimizer=QODE).set_name("MetaModelQODE", register=True)
NeuralMetaModelDE = ParametrizedMetaModel(algorithm="neural", multivariate_optimizer=DE).set_name(
    "NeuralMetaModelDE", register=True
)
SVMMetaModelDE = ParametrizedMetaModel(algorithm="svr", multivariate_optimizer=DE).set_name(
    "SVMMetaModelDE", register=True
)
RFMetaModelDE = ParametrizedMetaModel(algorithm="rf", multivariate_optimizer=DE).set_name(
    "RFMetaModelDE", register=True
)

MetaModelTwoPointsDE = ParametrizedMetaModel(multivariate_optimizer=TwoPointsDE).set_name(
    "MetaModelTwoPointsDE", register=True
)
NeuralMetaModelTwoPointsDE = ParametrizedMetaModel(
    algorithm="neural", multivariate_optimizer=TwoPointsDE
).set_name("NeuralMetaModelTwoPointsDE", register=True)
SVMMetaModelTwoPointsDE = ParametrizedMetaModel(algorithm="svr", multivariate_optimizer=TwoPointsDE).set_name(
    "SVMMetaModelTwoPointsDE", register=True
)
RFMetaModelTwoPointsDE = ParametrizedMetaModel(algorithm="rf", multivariate_optimizer=TwoPointsDE).set_name(
    "RFMetaModelTwoPointsDE", register=True
)


def rescaled(n: int, o: tp.Any):
    return Rescaled(base_optimizer=o, scale=1.0 / np.exp(np.log(n) * np.random.rand()))  # type: ignore


@registry.register
class MultiBFGSPlus(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        optims: tp.List[base.Optimizer] = []
        optims += [
            rescaled(num_workers, BFGS)(self.parametrization, num_workers=1) for _ in range(num_workers)
        ]
        for opt in optims[2:]:  # make sure initializations differ
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)


@registry.register
class LogMultiBFGSPlus(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        if budget is not None:
            num_workers = int(max(num_workers, 1 + np.log(budget)))
        optims: tp.List[base.Optimizer] = []
        optims += [
            rescaled(num_workers, BFGS)(self.parametrization, num_workers=1) for _ in range(num_workers)
        ]
        for opt in optims[2:]:  # make sure initializations differ
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)


@registry.register
class SqrtMultiBFGSPlus(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        if budget is not None:
            num_workers = int(max(num_workers, 1 + np.sqrt(budget)))
        optims: tp.List[base.Optimizer] = []
        optims += [
            rescaled(num_workers, BFGS)(self.parametrization, num_workers=1) for _ in range(num_workers)
        ]
        for opt in optims[2:]:  # make sure initializations differ
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)


@registry.register
class MultiCobylaPlus(Portfolio):
    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        optims: tp.List[base.Optimizer] = []
        optims += [
            rescaled(num_workers, Cobyla)(self.parametrization, num_workers=1) for _ in range(num_workers)
        ]
        for opt in optims[2:]:  # make sure initializations differ
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)


@registry.register
class MultiSQPPlus(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        optims: tp.List[base.Optimizer] = []
        optims += [
            rescaled(num_workers, SQP)(self.parametrization, num_workers=1) for _ in range(num_workers)
        ]
        for opt in optims[2:]:  # make sure initializations differ
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)


@registry.register
class BFGSCMAPlus(Portfolio):
    """Passive portfolio of MetaCMA and many BFGS."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        cma_workers = max(1, num_workers // 2)
        optims: tp.List[base.Optimizer] = [
            MetaCMA(self.parametrization, budget=budget, num_workers=cma_workers)
        ]
        optims += [
            rescaled(num_workers, BFGS)(self.parametrization, num_workers=1)
            for _ in range(num_workers - cma_workers)
        ]
        for opt in optims[2:]:  # make sure initializations differ
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)


@registry.register
class LogBFGSCMAPlus(Portfolio):
    """Passive portfolio of MetaCMA and many BFGS."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        if budget is not None:
            num_workers = int(max(num_workers, 1 + np.log(budget)))
        cma_workers = max(1, num_workers // 2)
        optims: tp.List[base.Optimizer] = [
            MetaCMA(self.parametrization, budget=budget, num_workers=cma_workers)
        ]
        optims += [
            rescaled(num_workers, BFGS)(self.parametrization, num_workers=1)
            for _ in range(num_workers - cma_workers)
        ]
        for opt in optims[2:]:  # make sure initializations differ
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)


@registry.register
class SqrtBFGSCMAPlus(Portfolio):
    """Passive portfolio of MetaCMA and many BFGS."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        if budget is not None:
            num_workers = int(max(num_workers, 1 + np.sqrt(budget)))
        cma_workers = max(1, num_workers // 2)
        optims: tp.List[base.Optimizer] = [
            MetaCMA(self.parametrization, budget=budget, num_workers=cma_workers)
        ]
        optims += [
            rescaled(num_workers, BFGS)(self.parametrization, num_workers=1)
            for _ in range(num_workers - cma_workers)
        ]
        for opt in optims[2:]:  # make sure initializations differ
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)


@registry.register
class SQPCMAPlus(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        cma_workers = max(1, num_workers // 2)
        optims: tp.List[base.Optimizer] = [
            MetaCMA(self.parametrization, budget=budget, num_workers=cma_workers)
        ]
        optims += [
            rescaled(num_workers, SQP)(self.parametrization, num_workers=1)
            for _ in range(num_workers - cma_workers)
        ]
        for opt in optims[2:]:  # make sure initializations differ
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)


@registry.register
class LogSQPCMAPlus(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        if budget is not None:
            num_workers = int(max(num_workers, 1 + np.log(budget)))
        cma_workers = max(1, num_workers // 2)
        optims: tp.List[base.Optimizer] = [
            MetaCMA(self.parametrization, budget=budget, num_workers=cma_workers)
        ]
        optims += [
            rescaled(num_workers, SQP)(self.parametrization, num_workers=1)
            for _ in range(num_workers - cma_workers)
        ]
        for opt in optims[2:]:  # make sure initializations differ
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)


@registry.register
class SqrtSQPCMAPlus(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        if budget is not None:
            num_workers = int(max(num_workers, 1 + np.sqrt(budget)))
        cma_workers = max(1, num_workers // 2)
        optims: tp.List[base.Optimizer] = [
            MetaCMA(self.parametrization, budget=budget, num_workers=cma_workers)
        ]
        optims += [
            rescaled(num_workers, SQP)(self.parametrization, num_workers=1)
            for _ in range(num_workers - cma_workers)
        ]
        for opt in optims[2:]:  # make sure initializations differ
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)


@registry.register
class MultiBFGS(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        optims: tp.List[base.Optimizer] = []
        optims += [RBFGS(self.parametrization, num_workers=1) for _ in range(num_workers)]
        for opt in optims[2:]:  # make sure initializations differ
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)


@registry.register
class LogMultiBFGS(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        if budget is not None:
            num_workers = int(max(num_workers, 1 + np.log(budget)))
        optims: tp.List[base.Optimizer] = []
        optims += [BFGS(self.parametrization, num_workers=1) for _ in range(num_workers)]
        for opt in optims[2:]:  # make sure initializations differ
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)


@registry.register
class SqrtMultiBFGS(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        if budget is not None:
            num_workers = int(max(num_workers, 1 + np.sqrt(budget)))
        optims: tp.List[base.Optimizer] = []
        optims += [BFGS(self.parametrization, num_workers=1) for _ in range(num_workers)]
        for opt in optims[2:]:  # make sure initializations differ
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)


@registry.register
class MultiCobyla(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        optims: tp.List[base.Optimizer] = []
        optims += [Cobyla(self.parametrization, num_workers=1) for _ in range(num_workers)]
        for opt in optims[2:]:  # make sure initializations differ
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)


@registry.register
class ForceMultiCobyla(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        optims: tp.List[base.Optimizer] = []
        force_num_workers = max(num_workers, int(np.sqrt(budget))) if budget is not None else num_workers
        optims += [Cobyla(self.parametrization, num_workers=1) for _ in range(force_num_workers)]
        for opt in optims[2:]:  # make sure initializations differ
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)


@registry.register
class MultiSQP(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        optims: tp.List[base.Optimizer] = []
        optims += [SQP(self.parametrization, num_workers=1) for _ in range(num_workers)]
        for opt in optims[2:]:  # make sure initializations differ
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)


@registry.register
class BFGSCMA(Portfolio):
    """Passive portfolio of MetaCMA and many BFGS."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        cma_workers = max(1, num_workers // 2)
        optims: tp.List[base.Optimizer] = [
            MetaCMA(self.parametrization, budget=budget, num_workers=cma_workers)
        ]
        optims += [BFGS(self.parametrization, num_workers=1) for _ in range(num_workers - cma_workers)]
        for opt in optims[2:]:  # make sure initializations differ
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)


@registry.register
class LogBFGSCMA(Portfolio):
    """Passive portfolio of MetaCMA and many BFGS."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        if budget is not None:
            num_workers = int(max(num_workers, 1 + np.log(budget)))
        cma_workers = max(1, num_workers // 2)
        optims: tp.List[base.Optimizer] = [
            MetaCMA(self.parametrization, budget=budget, num_workers=cma_workers)
        ]
        optims += [BFGS(self.parametrization, num_workers=1) for _ in range(num_workers - cma_workers)]
        for opt in optims[2:]:  # make sure initializations differ
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)


@registry.register
class SqrtBFGSCMA(Portfolio):
    """Passive portfolio of MetaCMA and many BFGS."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        if budget is not None:
            num_workers = int(max(num_workers, 1 + np.sqrt(budget)))
        cma_workers = max(1, num_workers // 2)
        optims: tp.List[base.Optimizer] = [
            MetaCMA(self.parametrization, budget=budget, num_workers=cma_workers)
        ]
        optims += [BFGS(self.parametrization, num_workers=1) for _ in range(num_workers - cma_workers)]
        for opt in optims[2:]:  # make sure initializations differ
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)


@registry.register
class SQPCMA(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        cma_workers = max(1, num_workers // 2)
        optims: tp.List[base.Optimizer] = [
            MetaCMA(self.parametrization, budget=budget, num_workers=cma_workers)
        ]
        optims += [SQP(self.parametrization, num_workers=1) for _ in range(num_workers - cma_workers)]
        for opt in optims[2:]:  # make sure initializations differ
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)


@registry.register
class LogSQPCMA(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        if budget is not None:
            num_workers = int(max(num_workers, 1 + np.log(budget)))
        cma_workers = max(1, num_workers // 2)
        optims: tp.List[base.Optimizer] = [
            MetaCMA(self.parametrization, budget=budget, num_workers=cma_workers)
        ]
        optims += [SQP(self.parametrization, num_workers=1) for _ in range(num_workers - cma_workers)]
        for opt in optims[2:]:  # make sure initializations differ
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)


@registry.register
class SqrtSQPCMA(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        if budget is not None:
            num_workers = int(max(num_workers, 1 + np.sqrt(budget)))
        cma_workers = max(1, num_workers // 2)
        optims: tp.List[base.Optimizer] = [
            MetaCMA(self.parametrization, budget=budget, num_workers=cma_workers)
        ]
        optims += [SQP(self.parametrization, num_workers=1) for _ in range(num_workers - cma_workers)]
        for opt in optims[2:]:  # make sure initializations differ
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)


@registry.register
class FSQPCMA(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        num_workers = max(num_workers, int(np.sqrt(budget)))  # type: ignore
        cma_workers = max(1, num_workers // 2)
        optims: tp.List[base.Optimizer] = [
            MetaCMA(self.parametrization, budget=budget, num_workers=cma_workers)
        ]
        optims += [SQP(self.parametrization, num_workers=1) for _ in range(num_workers - cma_workers)]
        for opt in optims[2:]:  # make sure initializations differ
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)


@registry.register
class F2SQPCMA(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        num_workers = max(num_workers, int(np.log(budget)))  # type: ignore
        cma_workers = max(1, num_workers // 2)
        optims: tp.List[base.Optimizer] = [
            MetaCMA(self.parametrization, budget=budget, num_workers=cma_workers)
        ]
        optims += [SQP(self.parametrization, num_workers=1) for _ in range(num_workers - cma_workers)]
        for opt in optims[2:]:  # make sure initializations differ
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)


@registry.register
class F3SQPCMA(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        num_workers = max(num_workers, int(np.sqrt(budget)))  # type: ignore
        cma_workers = max(1, num_workers // 2)
        optims: tp.List[base.Optimizer] = [
            MetaCMA(self.parametrization, budget=budget, num_workers=cma_workers)
        ]
        optims += [
            Rescaled(base_optimizer=SQP, scale=max(0.01, np.exp(-1.0 / np.random.rand())))(
                self.parametrization, num_workers=1
            )
            for _ in range(num_workers - cma_workers)
        ]
        for opt in optims[2:]:  # make sure initializations differ
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore
        self.optims.clear()
        self.optims.extend(optims)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)


@registry.register
class MultiDiscrete(Portfolio):
    """Combining 3 Discrete(1+1) optimizers. Active selection at 1/4th of the budget."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(
            parametrization, budget=budget, num_workers=num_workers, config=ConfPortfolio(warmup_ratio=0.25)
        )
        assert budget is not None
        self.optims.clear()
        self.optims = [
            DiscreteOnePlusOne(self.parametrization, budget=budget // 12, num_workers=num_workers),
            DiscreteBSOOnePlusOne(self.parametrization, budget=budget // 12, num_workers=num_workers),
            DoubleFastGADiscreteOnePlusOne(
                self.parametrization, budget=(budget // 4) - 2 * (budget // 12), num_workers=num_workers
            ),
        ]


@registry.register
class CMandAS2(Portfolio):
    """Competence map, with algorithm selection in one of the cases (3 CMAs)."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        # we need to manually create the parametrization if it's an int, so as to make sure
        # it is shared through instances
        optims: tp.List[base.OptCls] = [TwoPointsDE]
        if isinstance(parametrization, int):
            parametrization = p.Array(shape=(parametrization,))
        dim = parametrization.dimension
        assert budget is not None
        warmup_ratio = 2.0
        if budget < 201:
            optims = [OnePlusOne]
        if budget > 50 * dim or num_workers < 30:
            optims = [MetaModel for _ in range(3)]
            warmup_ratio = 0.1
        super().__init__(
            parametrization,
            budget=budget,
            num_workers=num_workers,
            config=ConfPortfolio(optimizers=optims, warmup_ratio=warmup_ratio),
        )


@registry.register
class CMandAS3(Portfolio):
    """Competence map, with algorithm selection in one of the cases (3 CMAs)."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        optims: tp.List[base.OptCls] = [TwoPointsDE]
        warmup_ratio = 2.0
        if isinstance(parametrization, int):
            parametrization = p.Array(shape=(parametrization,))
        dim = parametrization.dimension
        assert budget is not None
        if budget < 201:
            optims = [OnePlusOne]
        if budget > 50 * dim or num_workers < 30:
            if num_workers == 1:
                optims = [ChainCMAPowell for _ in range(3)]
            else:
                optims = [MetaCMA for _ in range(3)]
            warmup_ratio = 0.1
        super().__init__(
            parametrization,
            budget=budget,
            num_workers=num_workers,
            config=ConfPortfolio(optimizers=optims, warmup_ratio=warmup_ratio),
        )


@registry.register
class CM(Portfolio):
    """Competence map, simplest."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        optims: tp.List[base.OptCls] = [TwoPointsDE]
        if isinstance(parametrization, int):
            parametrization = p.Array(shape=(parametrization,))
        dim = parametrization.dimension
        assert budget is not None
        warmup_ratio = 2.0
        assert budget is not None
        if budget < 201:
            optims = [OnePlusOne]
        if budget > 50 * dim:
            optims = [MetaCMA]
        super().__init__(
            parametrization,
            budget=budget,
            num_workers=num_workers,
            config=ConfPortfolio(optimizers=optims, warmup_ratio=warmup_ratio),
        )


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


try:

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
            self._normalizer = p.helpers.Normalizer(self.parametrization)
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
                if self._init_budget is None:
                    assert self.budget is not None
                    init_budget = int(np.sqrt(self.budget))
                else:
                    init_budget = self._init_budget
                init_budget = max(2, init_budget)
                if self.gp_parameters is not None:
                    self._bo.set_gp_params(**self.gp_parameters)
                # init
                if self._middle_point:
                    self._bo.probe([0.5] * self.dimension, lazy=True)
                    init_budget -= 1
                if self._InitOpt is not None and init_budget > 0:
                    param = p.Array(shape=(self.dimension,)).set_bounds(lower=0, upper=1)
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
            data = self._normalizer.backward(np.asarray(x_probe))
            candidate = self.parametrization.spawn_child().set_standardized_data(data)
            candidate._meta["x_probe"] = x_probe
            return candidate

        def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
            if "x_probe" in candidate._meta:
                y = candidate._meta["x_probe"]
            else:
                data = candidate.get_standardized_data(reference=self.parametrization)
                y = self._normalizer.forward(data)  # tell not asked
            self._fake_function.register(y, -loss)  # minimizing
            self.bo.probe(y, lazy=False)
            # for some unknown reasons, BO wants to evaluate twice the same point,
            # but since it keeps a cache of the values, the registered value is not used
            # so we should clean the "fake" function
            self._fake_function._registered.clear()

        def _internal_provide_recommendation(self) -> tp.Optional[tp.ArrayLike]:
            if not self.archive:
                return None
            return self._normalizer.backward(
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
    BOSplit = ConfSplitOptimizer(max_num_vars=15, progressive=False, multivariate_optimizer=BO).set_name(
        "BOSplit", register=True
    )

except NameError:
    pass  # bayes_opt not available


class _BayesOptim(base.Optimizer):
    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        *,
        config: tp.Optional["BayesOptim"] = None,
    ) -> None:
        self._config = BayesOptim() if config is None else config
        cfg = self._config
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._normalizer = p.helpers.Normalizer(self.parametrization)

        from bayes_optim import RealSpace
        from bayes_optim.surrogate import GaussianProcess

        lb, ub = 1e-7, 1 - 1e-7
        space = RealSpace([lb, ub]) * self.dimension

        self._buffer: tp.List[float] = []
        self._newX: tp.List[float] = []
        self._losses: tp.List[float] = []

        # Setting DoE size as a percentage of the total budget if prop_doe_factor is not None
        init_budget = cfg.init_budget
        if cfg.prop_doe_factor and budget is not None:
            init_budget = round(cfg.prop_doe_factor * budget) if budget >= 10 else 5

        if cfg.pca:
            from bayes_optim.extension import PCABO as PcaBO

            self._alg = PcaBO(
                search_space=space,
                obj_fun=None,  # Assuming that this is not used :-)
                DoE_size=init_budget if init_budget is not None else 5,
                max_FEs=budget,
                verbose=True,
                n_point=1,  # We start with a sequential procedure, maybe we'll extend in a second moment
                n_components=cfg.n_components,
                acquisition_optimization={"optimizer": "BFGS"},
            )
        else:
            from bayes_optim import BO as BayesOptimBO_

            # hyperparameters of the GPR model
            thetaL = 1e-10 * (ub - lb) * np.ones(self.dimension)
            thetaU = 10 * (ub - lb) * np.ones(self.dimension)
            model = GaussianProcess(thetaL=thetaL, thetaU=thetaU)  # create the GPR model

            self._alg = BayesOptimBO_(
                search_space=space,
                obj_fun=None,
                model=model,
                DoE_size=init_budget if init_budget is not None else 5,
                max_FEs=budget,
                verbose=True,
            )

    def _internal_ask_candidate(self) -> p.Parameter:
        if not self._buffer:
            candidate = self._alg.ask()
            if not isinstance(candidate, list):
                candidate = candidate.tolist()
            self._buffer = candidate
        x_probe = self._buffer.pop()
        data = self._normalizer.backward(np.array(x_probe))
        candidate = self.parametrization.spawn_child().set_standardized_data(data)
        candidate._meta["x_probe"] = x_probe
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        self._newX.append(candidate._meta["x_probe"])
        self._losses.append(loss)
        if not self._buffer:
            if "x_probe" in candidate._meta:
                self._alg.tell(self._newX, self._losses)
            else:
                data = candidate.get_standardized_data(reference=self.parametrization)
                # Tell not asked:
                self._alg.tell(self._normalizer.forward(data), loss)
            self._newX = []
            self._losses = []

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        raise errors.TellNotAskedNotSupportedError


class BayesOptim(base.ConfiguredOptimizer):
    """
    Algorithms from bayes-optim package.

    We use:
    - BO
    - PCA-BO: Principle Component Analysis (PCA) Bayesian Optimization for dimensionality reduction in BO

    References

    [RaponiWB+20]
        Raponi, Elena, Hao Wang, Mariusz Bujny, Simonetta Boria, and Carola Doerr.
        "High dimensional bayesian optimization assisted by principal component analysis."
        In International Conference on Parallel Problem Solving from Nature, pp. 169-183.
        Springer, Cham, 2020.


    Parameters
    ----------
    init_budget: int or None
        Number of initialization algorithm steps
    pca: bool
        whether to use the PCA transformation defining PCA-BO rather than BO
    n_components: float or 0.95
        Principal axes in feature space, representing the directions of maximum variance in the data.
        It represents the percentage of explained variance
    prop_doe_factor: float or None
        Percentage of the initial budget used for DoE and eventually overwriting init_budget
    """

    no_parallelization = True
    recast = True

    # pylint: disable=unused-argument
    def __init__(
        self,
        *,
        init_budget: tp.Optional[int] = None,
        pca: tp.Optional[bool] = False,
        n_components: tp.Optional[float] = 0.95,
        prop_doe_factor: tp.Optional[float] = None,
    ) -> None:
        super().__init__(_BayesOptim, locals(), as_config=True)
        self.init_budget = init_budget
        self.pca = pca
        self.n_components = n_components
        self.prop_doe_factor = prop_doe_factor


PCABO = BayesOptim(pca=True).set_name("PCABO", register=True)
BayesOptimBO = BayesOptim().set_name("BayesOptimBO", register=True)


class _Chain(base.Optimizer):
    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        *,
        optimizers: tp.Optional[
            tp.Sequence[tp.Union[base.ConfiguredOptimizer, tp.Type[base.Optimizer]]]
        ] = None,
        budgets: tp.Sequence[tp.Union[str, int]] = (10,),
        no_crossing: tp.Optional[bool] = False,
    ) -> None:
        if optimizers is None:
            optimizers = [LHSSearch, DE]
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.no_crossing = no_crossing
        # delayed initialization
        # Either we have the budget for each algorithm, or the last algorithm uses the rest of the budget, so:
        self.optimizers: tp.List[base.Optimizer] = []
        converter = {
            "num_workers": self.num_workers,
            "dimension": self.dimension,
            "half": self.budget // 2 if self.budget else self.num_workers,
            "third": self.budget // 3 if self.budget else self.num_workers,
            "fourth": self.budget // 4 if self.budget else self.num_workers,
            "tenth": self.budget // 10 if self.budget else self.num_workers,
            "equal": self.budget // (len(budgets) + 1) if self.budget else self.num_workers,
            "most": (self.budget * 4) // 5 if self.budget else self.num_workers,
            "sqrt": int(np.sqrt(self.budget)) if self.budget else self.num_workers,
        }
        self.budgets = [max(1, converter[b]) if isinstance(b, str) else b for b in budgets]
        last_budget = None if self.budget is None else max(4, self.budget - sum(self.budgets))
        assert len(optimizers) == len(self.budgets) + 1
        assert all(
            x in ("equal", "fourth", "third", "half", "tenth", "dimension", "num_workers", "sqrt") or x > 0
            for x in self.budgets
        ), str(self.budgets)
        for opt, optbudget in zip(optimizers, self.budgets + [last_budget]):  # type: ignore
            self.optimizers.append(opt(self.parametrization, budget=optbudget, num_workers=self.num_workers))
        # print(f"Chaining of {self.optimizers}")
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
        chosen_index = 0
        for index, opt in enumerate(self.optimizers):
            sum_budget += float("inf") if opt.budget is None else opt.budget
            if self.num_ask < sum_budget:
                chosen_index = index
                break
        # if we are over budget, then use the last one...

        candidate = opt.ask()
        candidate._meta["optim_index"] = chosen_index
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        # Let us inform all concerned algorithms
        sum_budget = 0.0
        for k, opt in enumerate(self.optimizers):
            is_last = k == len(self.optimizers) - 1
            sum_budget += float("inf") if opt.budget is None or is_last else opt.budget
            if self.num_tell < sum_budget and not (
                self.no_crossing
                and ("optim_index" not in candidate._meta or k != candidate._meta["optim_index"])
            ):
                opt.tell(candidate, loss)

    def enable_pickling(self):
        for opt in self.optimizers:
            opt.enable_pickling()


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
        no_crossing: tp.Optional[bool] = False,
    ) -> None:
        super().__init__(_Chain, locals())


# new names
GeneticDE = Chaining([RotatedTwoPointsDE, TwoPointsDE], [200]).set_name(
    "GeneticDE", register=True
)  # Also known as CGDE
MemeticDE = Chaining([RotatedTwoPointsDE, TwoPointsDE, DE, SQP], ["fourth", "fourth", "fourth"]).set_name(
    "MemeticDE", register=True
)
QNDE = Chaining([QODE, RBFGS], ["half"]).set_name("QNDE", register=True)
ChainDE = Chaining([DE, RBFGS], ["half"]).set_name("ChainDE", register=True)
OpoDE = Chaining([OnePlusOne, QODE], ["half"]).set_name("OpoDE", register=True)
OpoTinyDE = Chaining([OnePlusOne, TinyQODE], ["half"]).set_name("OpoTinyDE", register=True)
QNDE.no_parallelization = True
ChainDE.no_parallelization = True
Carola1 = Chaining([Cobyla, MetaModel], ["half"]).set_name("Carola1", register=True)
Carola2 = Chaining([Cobyla, MetaModel, SQP], ["third", "third"]).set_name("Carola2", register=True)
DS2 = Chaining([Cobyla, MetaModelDSproba, SQP], ["third", "third"]).set_name("DS2", register=True)
Carola4 = Chaining([Cobyla, MetaModel, SQP], ["sqrt", "half"]).set_name("Carola4", register=True)
DS4 = Chaining([Cobyla, MetaModelDSproba, SQP], ["sqrt", "half"]).set_name("DS4", register=True)
Carola5 = Chaining([Cobyla, MetaModel, SQP], ["sqrt", "most"]).set_name("Carola5", register=True)
DS5 = Chaining([Cobyla, MetaModelDSproba, SQP], ["sqrt", "most"]).set_name("DS5", register=True)
Carola6 = Chaining([Cobyla, MetaModel, SQP], ["tenth", "most"]).set_name("Carola6", register=True)
DS6 = Chaining([Cobyla, MetaModelDSproba, SQP], ["tenth", "most"]).set_name("DS6", register=True)
DS6.no_parallelization = True
PCarola6 = Rescaled(
    base_optimizer=Chaining([Cobyla, MetaModel, SQP], ["tenth", "most"]), scale=10.0
).set_name("PCarola6", register=True)
pCarola6 = Rescaled(base_optimizer=Chaining([Cobyla, MetaModel, SQP], ["tenth", "most"]), scale=3.0).set_name(
    "pCarola6", register=True
)
Carola1.no_parallelization = True
Carola2.no_parallelization = True
DS2.no_parallelization = True
Carola4.no_parallelization = True
DS4.no_parallelization = True
DS5.no_parallelization = True
Carola5.no_parallelization = True
Carola6.no_parallelization = True
PCarola6.no_parallelization = True
pCarola6.no_parallelization = True
Carola7 = Chaining([MultiCobyla, MetaModel, MultiSQP], ["tenth", "most"]).set_name("Carola7", register=True)
Carola8 = Chaining([Cobyla, CmaFmin2, SQP], ["tenth", "most"]).set_name("Carola8", register=True)
DS8 = Chaining([Cobyla, DSproba, SQP], ["tenth", "most"]).set_name("DS8", register=True)
Carola9 = Chaining(
    [Cobyla, ParametrizedMetaModel(multivariate_optimizer=CmaFmin2), SQP], ["tenth", "most"]
).set_name("Carola9", register=True)
DS9 = Chaining(
    [Cobyla, ParametrizedMetaModel(multivariate_optimizer=DSproba), SQP], ["tenth", "most"]
).set_name("DS9", register=True)
Carola9.no_parallelization = True
Carola8.no_parallelization = True
DS8.no_parallelization = True
Carola10 = Chaining([Cobyla, CmaFmin2, RBFGS], ["tenth", "most"]).set_name("Carola10", register=True)
Carola10.no_parallelization = True


@registry.register
class Carola3(Portfolio):
    """Passive portfolio of MetaCMA and many SQP."""

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        cma_workers = max(1, num_workers // 2)
        optims: tp.List[base.Optimizer] = [
            MetaModel(self.parametrization, budget=budget, num_workers=cma_workers)
        ]
        optims += [Carola1(self.parametrization, num_workers=1) for _ in range(num_workers - cma_workers)]
        for opt in optims[2:]:  # make sure initializations differ
            opt.initial_guess = self._rng.normal(0, 1, self.dimension)  # type: ignore
        self.optims.clear()
        self.optims.extend(optims)


BAR = ConfPortfolio(optimizers=[OnePlusOne, DiagonalCMA, OpoDE], warmup_ratio=0.5).set_name(
    "BAR", register=True
)
BAR2 = ConfPortfolio(optimizers=[OnePlusOne, MetaCMA, OpoDE], warmup_ratio=0.5).set_name(
    "BAR2", register=True
)
BAR3 = ConfPortfolio(optimizers=[RandomSearch, OnePlusOne, MetaCMA, QNDE], warmup_ratio=0.5).set_name(
    "BAR3", register=True
)
MemeticDE.no_parallelization = True
discretememetic = Chaining(
    [RandomSearch, DiscreteLenglerOnePlusOne, DiscreteOnePlusOne], ["third", "third"]
).set_name("discretememetic", register=True)
# discretememeticT = Chaining(
#     [RandomSearch, DiscreteLenglerOnePlusOneT, DiscreteOnePlusOneT], ["tenth", "third"]
# ).set_name("discretememeticT", register=True)
ChainCMAPowell = Chaining([MetaCMA, Powell], ["half"]).set_name("ChainCMAPowell", register=True)
ChainDSPowell = Chaining([DSproba, Powell], ["half"]).set_name("ChainDSPowell", register=True)
ChainCMAPowell.no_parallelization = True  # TODO make this automatic
ChainDSPowell.no_parallelization = True  # TODO make this automatic
ChainMetaModelSQP = Chaining([MetaModel, SQP], ["half"]).set_name("ChainMetaModelSQP", register=True)
ChainMetaModelDSSQP = Chaining([MetaModelDSproba, SQP], ["half"]).set_name(
    "ChainMetaModelDSSQP", register=True
)
ChainMetaModelSQP.no_parallelization = True
ChainMetaModelDSSQP.no_parallelization = True
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
ChainNaiveTBPSACMAPowell = Chaining([NaiveTBPSA, MetaCMA, Powell], ["third", "third"]).set_name(
    "ChainNaiveTBPSACMAPowell", register=True
)
ChainNaiveTBPSACMAPowell.no_parallelization = True
BAR4 = ConfPortfolio(optimizers=[ChainMetaModelSQP, QNDE], warmup_ratio=0.5).set_name("BAR4", register=True)


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
                            axis=0 if self.isotropic else None,  # type: ignore
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
                    np.sum(stdd, axis=0 if self.isotropic else None)  # type: ignore
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
        self.has_real_noise = not funcinfo.deterministic
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
            self._optim = self._optim if not isinstance(self._optim, NGOptBase) else self._optim.optim
            logger.debug("%s selected %s optimizer.", *(x.name for x in (self, self._optim)))
        return self._optim

    def _select_optimizer_cls(self) -> base.OptCls:
        # pylint: disable=too-many-nested-blocks
        assert self.budget is not None
        if self.has_noise and self.has_discrete_not_softmax:
            # noise and discrete: let us merge evolution and bandits.
            cls: base.OptCls = DoubleFastGADiscreteOnePlusOne if self.dimension < 60 else CMA
        else:
            if self.has_real_noise and self.fully_continuous:
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
                                    DE
                                    if self.dimension > 2000
                                    else MetaCMA if self.dimension > 1 else OnePlusOne
                                )
        # print(f"NGOptbase: budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, {cls}")
        return cls

    def _internal_ask_candidate(self) -> p.Parameter:
        return self.optim.ask()

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        self.optim.tell(candidate, loss)

    def recommend(self) -> p.Parameter:
        return self.optim.recommend()

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        self.optim.tell(candidate, loss)

    def _info(self) -> tp.Dict[str, tp.Any]:
        out = {"sub-optim": self.optim.name}
        out.update(self.optim._info())  # this will work for recursive NGOpt calls
        return out

    def enable_pickling(self) -> None:
        self.optim.enable_pickling()


@registry.register
class NGOptDSBase(NGOptBase):
    """Nevergrad optimizer by competence map."""

    def _select_optimizer_cls(self) -> base.OptCls:
        # pylint: disable=too-many-nested-blocks
        assert self.budget is not None
        if self.has_noise and self.has_discrete_not_softmax:
            # noise and discrete: let us merge evolution and bandits.
            cls: base.OptCls = DoubleFastGADiscreteOnePlusOne if self.dimension < 60 else CMA
        else:
            if self.has_noise and self.fully_continuous:
                # This is the real of population control. FIXME: should we pair with a bandit ?
                cls = Chaining([SQOPSO, OptimisticDiscreteOnePlusOne], ["half"])
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
                            cls = ChainDSPowell
                        else:
                            if self.num_workers == 1 and self.budget < self.dimension * 30:
                                # One plus one so good in large ratio "dimension / budget".
                                cls = OnePlusOne if self.dimension > 30 else Cobyla
                            else:
                                # DE is great in such a case (?).
                                cls = (
                                    DE
                                    if self.dimension > 2000
                                    else MetaCMA if self.dimension > 1 else OnePlusOne
                                )
        # print(f"NGOptbase: budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, {cls}")
        return cls


@registry.register
class Shiwa(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

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
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

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
                                                and budget < 7 * self.dimension**2
                                            ):
                                                optimClass = DiagonalCMA
                                            elif (
                                                3 * num_workers > self.dimension**2
                                                and budget > self.dimension**2
                                            ):
                                                optimClass = MetaModel
                                            else:
                                                optimClass = CMA
        return optimClass


@registry.register
class NGOpt8(NGOpt4):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

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
                and self.num_workers < self.budget
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


class NGOpt12(NGOpt10):
    def _select_optimizer_cls(self) -> base.OptCls:
        cma_vars = max(1, 4 + int(3 * np.log(self.dimension)))
        if (
            not self.has_noise
            and self.fully_continuous
            and self.num_workers <= cma_vars
            and self.dimension < 100  # was 50 in 15, 16, 17
            and self.budget is not None
            and self.budget < self.dimension * 50
            and self.budget > min(50, self.dimension * 5)
        ):
            return MetaModel
            # return ChainMetaModelSQP
        elif (
            not self.has_noise
            and self.fully_continuous
            and self.num_workers <= cma_vars
            and self.dimension < 100  # was 50 in 15, 16, 17
            and self.budget is not None
            and self.budget < self.dimension * 5
            and self.budget > 50
        ):
            return MetaModel
        else:
            return super()._select_optimizer_cls()


class NGOpt13(NGOpt12):  # Also known as NGOpt12H
    def _select_optimizer_cls(self) -> base.OptCls:
        if (
            not self.has_noise
            and self.budget is not None
            and self.num_workers * 3 < self.budget
            and self.dimension < 8
            and self.budget < 80
        ):
            return HyperOpt
        else:
            return super()._select_optimizer_cls()


class NGOpt14(NGOpt12):  # Also known as NGOpt12H_nohyperopt
    def _select_optimizer_cls(self) -> base.OptCls:
        if self.budget is not None and self.budget < 600:
            return MetaModel
        else:
            return super()._select_optimizer_cls()


@registry.register
class NGOpt15(NGOpt12):
    def _select_optimizer_cls(self) -> base.OptCls:
        if (
            self.budget is not None
            and self.fully_continuous
            and self.budget < self.dimension**2 * 2
            and self.num_workers == 1
            and not self.has_noise
            and self.num_objectives < 2
        ):
            return MetaModelOnePlusOne  # OnePlusOne seems equivalent so far
        elif self.fully_continuous and self.budget is not None and self.budget < 600:
            return MetaModel
        else:
            return super()._select_optimizer_cls()


@registry.register
class NGOpt16(NGOpt15):
    def _select_optimizer_cls(self) -> base.OptCls:
        if (
            self.budget is not None
            and self.fully_continuous
            and self.budget < 200 * self.dimension
            and self.num_workers == 1
            and not self.has_noise
            and self.num_objectives < 2
            and p.helpers.Normalizer(self.parametrization).fully_bounded
        ):
            return Cobyla
        else:
            return super()._select_optimizer_cls()


class NGOpt21(NGOpt16):
    def _select_optimizer_cls(self) -> base.OptCls:
        cma_vars = max(1, 4 + int(3 * np.log(self.dimension)))
        num = 1 + (4 * self.budget) // (self.dimension * 1000) if self.budget is not None else 1
        if (
            self.budget is not None
            and self.budget > 500 * self.dimension
            and self.fully_continuous
            and self.num_workers <= num * cma_vars
        ):  # Discrete case ?
            return ConfPortfolio(
                optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.3**i) for i in range(num)],
                warmup_ratio=0.5,
            )
        else:
            return super()._select_optimizer_cls()


@registry.register
class NGOpt36(NGOpt16):
    def _select_optimizer_cls(self) -> base.OptCls:
        num = (
            1 + int(np.sqrt(4.0 * (4 * self.budget) // (self.dimension * 1000)))
            if self.budget is not None
            else 1
        )
        cma_vars = max(1, 4 + int(3 * np.log(self.dimension)))
        if (
            self.budget is not None
            and self.budget > 500 * self.dimension
            and self.fully_continuous
            and self.num_workers <= num * cma_vars
        ):  # Discrete case ?
            return ConfPortfolio(
                optimizers=[Rescaled(base_optimizer=NGOpt14, scale=0.9**i) for i in range(num)],
                warmup_ratio=0.5,
            )
        else:
            return super()._select_optimizer_cls()


class NGOpt38(NGOpt16):
    def _select_optimizer_cls(self) -> base.OptCls:
        # Special cases in the bounded case
        if (
            self.budget is not None
            and self.budget > 500 * self.dimension
            and self.fully_continuous
            and not self.has_noise
            and self.num_objectives < 2
            and self.num_workers == 1
            and p.helpers.Normalizer(self.parametrization).fully_bounded
        ):
            if (
                self.budget > 5000 * self.dimension
            ):  # Asymptotically let us trust NGOpt36 and its subtle restart.
                return NGOpt36
            if self.dimension < 5:  # Low dimension: let us hit the bounds.
                return NGOpt21
            if self.dimension < 10:  # Moderate dimension: reasonable restart + bet and run.
                num = 1 + int(np.sqrt(8.0 * (8 * self.budget) // (self.dimension * 1000)))
                return ConfPortfolio(optimizers=[NGOpt14] * num, warmup_ratio=0.7)
            if self.dimension < 20:  # Nobody knows why this seems to be so good.
                num = self.budget // (500 * self.dimension)
                return ConfPortfolio(
                    optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.3**i) for i in range(num)],
                    warmup_ratio=0.5,
                )
            # We need a special case for dim < 30 ---> let's see later.
            # Otherwise, let us go back to normal life: NGOpt16 which rocks in many cases, possibly Cobyla.
            return NGOpt16
        elif (  # This might be specific of high-precision cases.
            self.budget is not None
            and self.fully_continuous
            and not self.has_noise
            and self.num_objectives < 2
            and self.num_workers == 1
            and self.budget > 50 * self.dimension
            and p.helpers.Normalizer(self.parametrization).fully_bounded
        ):
            return NGOpt8 if self.dimension < 3 else NGOpt15
        else:
            return super()._select_optimizer_cls()


@registry.register
class NGOpt39(NGOpt16):
    def _select_optimizer_cls(self) -> base.OptCls:
        if self.fully_continuous and self.has_noise:  # In particular for neuro-DPS.
            DeterministicMix = ConfPortfolio(optimizers=[DiagonalCMA, PSO, GeneticDE])
            return Chaining([DeterministicMix, OptimisticNoisyOnePlusOne], ["half"])

        cma_vars = max(1, 4 + int(3 * np.log(self.dimension)))
        num36 = (
            1 + int(np.sqrt(4.0 * (4 * self.budget) // (self.dimension * 1000)))
            if self.budget is not None
            else 1
        )
        num21 = 1 + (4 * self.budget) // (self.dimension * 1000) if self.budget is not None else 1
        num_dim10 = (
            1 + int(np.sqrt(8.0 * (8 * self.budget) // (self.dimension * 1000)))
            if self.budget is not None
            else 1
        )
        num_dim20 = self.budget // (500 * self.dimension) if self.budget is not None else 1
        para = 1
        if self.budget is not None and self.budget > 5000 * self.dimension:
            para = num36 * cma_vars
        elif self.dimension < 5:
            para = num21 * cma_vars
        elif self.dimension < 10:
            para = num_dim10 * cma_vars
        elif self.dimension < 20:
            para = num_dim20 * cma_vars

        # Special cases in the bounded case
        if (
            self.budget is not None
            and self.budget > 500 * self.dimension
            and self.fully_continuous
            and not self.has_noise
            and self.num_objectives < 2
            and self.num_workers <= para
            and p.helpers.Normalizer(self.parametrization).fully_bounded
        ):
            if self.dimension == 1:
                return NGOpt16
            if (
                self.budget > 5000 * self.dimension
            ):  # Asymptotically let us trust NGOpt36 and its subtle restart.
                return NGOpt36
            if self.dimension < 5:  # Low dimension: let us hit the bounds.
                return NGOpt21
            if self.dimension < 10:  # Moderate dimension: reasonable restart + bet and run.
                num = 1 + int(np.sqrt(8.0 * (8 * self.budget) // (self.dimension * 1000)))
                return ConfPortfolio(optimizers=[NGOpt14] * num, warmup_ratio=0.7)
            if self.dimension < 20:  # Nobody knows why this seems to be so good.
                num = self.budget // (500 * self.dimension)
                return ConfPortfolio(
                    optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.3**i) for i in range(num)],
                    warmup_ratio=0.5,
                )
            if self.num_workers == 1:
                return CmaFmin2
            # We need a special case for dim < 30 ---> let's see later.
            # Otherwise, let us go back to normal life: NGOpt16 which rocks in many cases, possibly Cobyla.
            return NGOpt16
        elif (  # This might be specific of high-precision cases.
            self.budget is not None
            and self.fully_continuous
            and not self.has_noise
            and self.num_objectives < 2
            and self.num_workers <= cma_vars
            and self.budget > 50 * self.dimension
            and p.helpers.Normalizer(self.parametrization).fully_bounded
        ):
            if self.dimension < 3:
                return NGOpt8
            if self.dimension <= 20 and self.num_workers == 1:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            return NGOpt15
        else:
            return NGOpt16


@registry.register
class NGOptRW(NGOpt39):
    def _select_optimizer_cls(self) -> base.OptCls:
        if self.fully_continuous and not self.has_noise and self.budget >= 12 * self.dimension:  # type: ignore
            return ConfPortfolio(
                optimizers=[GeneticDE, PSO, super()._select_optimizer_cls()], warmup_ratio=0.33
            )
        else:
            return super()._select_optimizer_cls()


@registry.register
class NGOptF(NGOpt39):
    def _select_optimizer_cls(self) -> base.OptCls:

        best = defaultdict(lambda: defaultdict(list))  # type: ignore

        def recommend_method(d, nod):
            best[12][833.333] += ["MemeticDE"]
            best[12][83.3333] += ["MemeticDE"]
            best[12][8.33333] += ["RealSpacePSO"]
            best[12][0.833333] += ["CauchyOnePlusOne"]
            best[12][833.333] += ["VLPCMA"]
            best[12][83.3333] += ["NLOPT_LN_SBPLX"]
            best[12][8.33333] += ["NLOPT_LN_SBPLX"]
            best[12][0.833333] += ["NLOPT_LN_SBPLX"]
            best[12][833.333] += ["VLPCMA"]
            best[12][83.3333] += ["MemeticDE"]
            best[12][8.33333] += ["SMAC3"]
            best[12][0.833333] += ["Cobyla"]
            best[24][416.667] += ["VLPCMA"]
            best[24][41.6667] += ["Wiz"]
            best[24][4.16667] += ["NLOPT_LN_SBPLX"]
            best[24][0.416667] += ["Cobyla"]
            best[24][416.667] += ["NLOPT_LN_SBPLX"]
            best[24][41.6667] += ["Wiz"]
            best[24][4.16667] += ["NLOPT_LN_SBPLX"]
            best[24][0.416667] += ["NLOPT_LN_SBPLX"]
            best[24][416.667] += ["ChainDiagonalCMAPowell"]
            best[24][41.6667] += ["NLOPT_LN_SBPLX"]
            best[24][4.16667] += ["QORealSpacePSO"]
            best[24][0.416667] += ["Cobyla"]
            best[2][5000] += ["NGOpt16"]
            best[2][500] += ["LhsDE"]
            best[2][50] += ["SODE"]
            best[2][5] += ["Carola2"]
            best[2][5000] += ["MetaModelQODE"]
            best[2][500] += ["MetaModelQODE"]
            best[2][50] += ["PCABO"]
            best[2][5] += ["HammersleySearchPlusMiddlePoint"]
            best[2][5000] += ["QORealSpacePSO"]
            best[2][500] += ["ChainDiagonalCMAPowell"]
            best[2][50] += ["MetaModelQODE"]
            best[2][5] += ["Cobyla"]
            best[5][2000] += ["MemeticDE"]
            best[5][200] += ["MemeticDE"]
            best[5][20] += ["LhsDE"]
            best[5][2] += ["MultiSQP"]
            best[5][2000] += ["MemeticDE"]
            best[5][200] += ["LhsDE"]
            best[5][20] += ["LhsDE"]
            best[5][2] += ["NLOPT_LN_SBPLX"]
            best[5][2000] += ["LhsDE"]
            best[5][200] += ["VLPCMA"]
            best[5][20] += ["BayesOptimBO"]
            best[5][2] += ["NLOPT_LN_SBPLX"]
            best[96][0.104167] += ["PCABO"]
            bestdist = float("inf")
            for d1 in best:
                for nod2 in best[d1]:
                    dist = (d - d1) ** 2 + (nod - nod2) ** 2
                    if dist < bestdist:
                        bestdist = dist
                        bestalg = best[d1][nod2]
            return bestalg

        if self.fully_continuous and not self.has_noise:
            algs = recommend_method(self.dimension, self.budget / self.dimension)  # type: ignore
            if self.num_workers > 1:
                algs = [a for a in algs if not registry[a].no_parallelization]
                if len(algs) == 0:
                    return SQPCMA
            return ConfPortfolio(optimizers=[registry[a] for a in algs], warmup_ratio=0.6)

        if self.fully_continuous and not self.has_noise and self.budget >= 12 * self.dimension:  # type: ignore
            return ConfPortfolio(optimizers=[GeneticDE, PSO, NGOpt39], warmup_ratio=0.33)
        else:
            return super()._select_optimizer_cls()


@registry.register
class NGOptF2(NGOpt39):
    def _select_optimizer_cls(self) -> base.OptCls:

        best = defaultdict(lambda: defaultdict(list))  # type: ignore

        def recommend_method(d, nod):
            best[12][833.333] += ["MemeticDE"]
            best[12][83.3333] += ["MemeticDE"]
            best[12][8.33333] += ["RealSpacePSO"]
            best[12][0.833333] += ["CauchyOnePlusOne"]
            best[12][833.333] += ["VLPCMA"]
            best[12][83.3333] += ["NLOPT_LN_SBPLX"]
            best[12][8.33333] += ["NLOPT_LN_SBPLX"]
            best[12][0.833333] += ["NLOPT_LN_SBPLX"]
            best[12][833.333] += ["VLPCMA"]
            best[12][83.3333] += ["MemeticDE"]
            best[12][8.33333] += ["SMAC3"]
            best[12][0.833333] += ["Cobyla"]
            best[24][416.667] += ["VLPCMA"]
            best[24][41.6667] += ["Wiz"]
            best[24][4.16667] += ["NLOPT_LN_SBPLX"]
            best[24][0.416667] += ["Cobyla"]
            best[24][416.667] += ["NLOPT_LN_SBPLX"]
            best[24][41.6667] += ["Wiz"]
            best[24][4.16667] += ["NLOPT_LN_SBPLX"]
            best[24][0.416667] += ["NLOPT_LN_SBPLX"]
            best[24][416.667] += ["ChainDiagonalCMAPowell"]
            best[24][41.6667] += ["NLOPT_LN_SBPLX"]
            best[24][4.16667] += ["QORealSpacePSO"]
            best[24][0.416667] += ["Cobyla"]
            best[2][5000] += ["NGOpt16"]
            best[2][500] += ["LhsDE"]
            best[2][50] += ["SODE"]
            best[2][5] += ["Carola2"]
            best[2][5000] += ["MetaModelQODE"]
            best[2][500] += ["MetaModelQODE"]
            best[2][50] += ["PCABO"]
            best[2][5] += ["HammersleySearchPlusMiddlePoint"]
            best[2][5000] += ["QORealSpacePSO"]
            best[2][500] += ["ChainDiagonalCMAPowell"]
            best[2][50] += ["MetaModelQODE"]
            best[2][5] += ["Cobyla"]
            best[5][2000] += ["MemeticDE"]
            best[5][200] += ["MemeticDE"]
            best[5][20] += ["LhsDE"]
            best[5][2] += ["MultiSQP"]
            best[5][2000] += ["MemeticDE"]
            best[5][200] += ["LhsDE"]
            best[5][20] += ["LhsDE"]
            best[5][2] += ["NLOPT_LN_SBPLX"]
            best[5][2000] += ["LhsDE"]
            best[5][200] += ["VLPCMA"]
            best[5][20] += ["BayesOptimBO"]
            best[5][2] += ["NLOPT_LN_SBPLX"]
            best[96][0.104167] += ["PCABO"]
            bestdist = float("inf")
            for d1 in best:
                for nod2 in best[d1]:
                    dist = (d - d1) ** 2 + (nod - nod2) ** 2
                    if dist < bestdist:
                        bestdist = dist
                        bestalg = best[d1][nod2]
            return bestalg

        if self.fully_continuous and not self.has_noise:
            algs = recommend_method(self.dimension, self.budget / self.dimension)  # type: ignore
            if self.num_workers > 1:
                algs = [a for a in algs if not registry[a].no_parallelization]
                if len(algs) == 0:
                    return SQPCMA

            def most_frequent(List):
                return max(set(List), key=List.count)

            return registry[most_frequent(algs)]

        if self.fully_continuous and not self.has_noise and self.budget >= 12 * self.dimension:  # type: ignore
            return ConfPortfolio(optimizers=[GeneticDE, PSO, NGOpt39], warmup_ratio=0.33)
        else:
            return super()._select_optimizer_cls()


@registry.register
class NGOptF3(NGOpt39):
    def _select_optimizer_cls(self) -> base.OptCls:

        best = defaultdict(lambda: defaultdict(list))  # type: ignore

        def recommend_method(d, nod):
            best[12][833.333] += ["MemeticDE"]
            best[12][83.3333] += ["NGOptRW"]
            best[12][8.33333] += ["RealSpacePSO"]
            best[12][0.833333] += ["ASCMADEthird"]
            best[12][8333.33] += ["GeneticDE"]
            best[12][833.333] += ["TripleCMA"]
            best[12][83.3333] += ["NLOPT_LN_SBPLX"]
            best[12][8.33333] += ["NLOPT_LN_SBPLX"]
            best[12][0.833333] += ["NLOPT_LN_SBPLX"]
            best[12][8333.33] += ["GeneticDE"]
            best[12][833.333] += ["VLPCMA"]
            best[12][83.3333] += ["MemeticDE"]
            best[12][8.33333] += ["SMAC3"]
            best[12][0.833333] += ["Cobyla"]
            best[24][416.667] += ["VLPCMA"]
            best[24][41.6667] += ["Wiz"]
            best[24][4.16667] += ["NLOPT_LN_SBPLX"]
            best[24][0.416667] += ["Cobyla"]
            best[24][4166.67] += ["NGOptF"]
            best[24][416.667] += ["NLOPT_LN_SBPLX"]
            best[24][41.6667] += ["ChainNaiveTBPSAPowell"]
            best[24][4.16667] += ["NLOPT_LN_SBPLX"]
            best[24][0.416667] += ["NLOPT_LN_NELDERMEAD"]
            best[24][4166.67] += ["Carola1"]
            best[24][416.667] += ["ChainDiagonalCMAPowell"]
            best[24][41.6667] += ["NLOPT_LN_SBPLX"]
            best[24][4.16667] += ["NLOPT_GN_CRS2_LM"]
            best[24][0.416667] += ["Cobyla"]
            best[2][500000] += ["NGOptF2"]
            best[2][50000] += ["NeuralMetaModelDE"]
            best[2][5000] += ["ASCMADEthird"]
            best[2][500] += ["LQODE"]
            best[2][50] += ["SODE"]
            best[2][5] += ["Carola2"]
            best[2][500000] += ["NLOPT_GN_CRS2_LM"]
            best[2][50000] += ["NGOptF2"]
            best[2][5000] += ["MetaModelQODE"]
            best[2][500] += ["MetaModelQODE"]
            best[2][50] += ["PCABO"]
            best[2][5] += ["HammersleySearchPlusMiddlePoint"]
            best[2][500000] += ["NGOptF"]
            best[2][50000] += ["NGOptF"]
            best[2][5000] += ["LQODE"]
            best[2][500] += ["ChainDiagonalCMAPowell"]
            best[2][50] += ["NLOPT_GN_DIRECT"]
            best[2][5] += ["Cobyla"]
            best[48][20.8333] += ["RLSOnePlusOne"]
            best[48][2.08333] += ["NGOptF"]
            best[48][0.208333] += ["NGOptF2"]
            best[48][208.333] += ["MetaModelQODE"]
            best[48][20.8333] += ["DiscreteLengler2OnePlusOne"]
            best[48][2.08333] += ["NGOptF"]
            best[48][0.208333] += ["NGOptF2"]
            best[48][2083.33] += ["NGOptF2"]
            best[48][208.333] += ["ChainNaiveTBPSAPowell"]
            best[48][20.8333] += ["ChainNaiveTBPSAPowell"]
            best[48][2.08333] += ["NLOPT_LN_NELDERMEAD"]
            best[48][0.208333] += ["BOBYQA"]
            best[5][20000] += ["NGOptF"]
            best[5][2000] += ["MemeticDE"]
            best[5][200] += ["TwoPointsDE"]
            best[5][20] += ["NGOptF2"]
            best[5][2] += ["MultiSQP"]
            best[5][20000] += ["CmaFmin2"]
            best[5][2000] += ["DiscreteDE"]
            best[5][200] += ["LhsDE"]
            best[5][20] += ["SMAC3"]
            best[5][2] += ["NLOPT_LN_SBPLX"]
            best[5][200000] += ["LQODE"]
            best[5][20000] += ["NGOptF"]
            best[5][2000] += ["LhsDE"]
            best[5][200] += ["FCMA"]
            best[5][20] += ["SMAC3"]
            best[5][2] += ["NLOPT_LN_SBPLX"]
            best[96][10.4167] += ["Carola2"]
            best[96][1.04167] += ["ChainDiagonalCMAPowell"]
            best[96][0.104167] += ["MetaModelQODE"]
            best[96][10.4167] += ["NGOpt8"]
            best[96][1.04167] += ["ChoiceBase"]
            best[96][0.104167] += ["Carola1"]
            best[96][104.167] += ["NGOptF"]
            best[96][10.4167] += ["CMandAS3"]
            best[96][1.04167] += ["ASCMADEthird"]
            best[96][0.104167] += ["CMAtuning"]
            bestdist = float("inf")
            for d1 in best:
                for nod2 in best[d1]:
                    dist = (d - d1) ** 2 + (nod - nod2) ** 2
                    if dist < bestdist:
                        bestdist = dist
                        bestalg = best[d1][nod2]
            return bestalg

        if self.fully_continuous and not self.has_noise:
            algs = recommend_method(self.dimension, self.budget / self.dimension)  # type: ignore
            if self.num_workers > 1:
                algs = [a for a in algs if not registry[a].no_parallelization]
                if len(algs) == 0:
                    return SQPCMA

            def most_frequent(List):
                return max(set(List), key=List.count)

            return registry[most_frequent(algs)]

        if self.fully_continuous and not self.has_noise and self.budget >= 12 * self.dimension:  # type: ignore
            return ConfPortfolio(optimizers=[GeneticDE, PSO, NGOpt39], warmup_ratio=0.33)
        else:
            return super()._select_optimizer_cls()


@registry.register
class NGOptF5(NGOpt39):
    def _select_optimizer_cls(self) -> base.OptCls:

        best = defaultdict(lambda: defaultdict(list))  # type: ignore

        def recommend_method(d, nod):
            best[12][8333.33] += ["DiscreteDE"]
            best[12][833.333] += ["DiscreteDE"]
            best[12][83.3333] += ["MetaModelDE"]
            best[12][8.33333] += ["NGOpt"]
            best[12][0.833333] += ["NLOPT_GN_DIRECT_L"]
            best[12][83333.3] += ["RPowell"]
            best[12][8333.33] += ["DiscreteDE"]
            best[12][833.333] += ["NLOPT_GN_ISRES"]
            best[12][83.3333] += ["SMAC3"]
            best[12][8.33333] += ["NGOpt"]
            best[12][0.833333] += ["NLOPT_LN_SBPLX"]
            best[12][83333.3] += ["NGOptBase"]
            best[12][8333.33] += ["RPowell"]
            best[12][833.333] += ["ChainMetaModelPowell"]
            best[12][83.3333] += ["RotatedTwoPointsDE"]
            best[12][8.33333] += ["NGOpt39"]
            best[12][0.833333] += ["NLOPT_LN_NEWUOA_BOUND"]
            best[24][4166.67] += ["CmaFmin2"]
            best[24][416.667] += ["CmaFmin2"]
            best[24][41.6667] += ["CmaFmin2"]
            best[24][4.16667] += ["NLOPT_LN_SBPLX"]
            best[24][0.416667] += ["Cobyla"]
            best[24][41666.7] += ["MultiCMA"]
            best[24][4166.67] += ["CmaFmin2"]
            best[24][416.667] += ["RotatedTwoPointsDE"]
            best[24][41.6667] += ["CmaFmin2"]
            best[24][4.16667] += ["pysot"]
            best[24][0.416667] += ["NLOPT_LN_NELDERMEAD"]
            best[24][41666.7] += ["MultiCMA"]
            best[24][4166.67] += ["Shiwa"]
            best[24][416.667] += ["CmaFmin2"]
            best[24][41.6667] += ["NLOPT_GN_CRS2_LM"]
            best[24][4.16667] += ["NLOPT_GN_CRS2_LM"]
            best[24][0.416667] += ["Cobyla"]
            best[2][500000] += ["BAR"]
            best[2][50000] += ["ASCMADEthird"]
            best[2][5000] += ["ChainMetaModelPowell"]
            best[2][500] += ["DiscreteDE"]
            best[2][50] += ["RFMetaModelOnePlusOne"]
            best[2][5] += ["Carola2"]
            best[2][500000] += ["BAR3"]
            best[2][50000] += ["ChainCMAPowell"]
            best[2][5000] += ["ASCMADEthird"]
            best[2][500] += ["NeuralMetaModelTwoPointsDE"]
            best[2][50] += ["CMandAS2"]
            best[2][5] += ["NaiveTBPSA"]
            best[2][500000] += ["BAR"]
            best[2][50000] += ["ASCMADEthird"]
            best[2][5000] += ["CM"]
            best[2][500] += ["ChainDiagonalCMAPowell"]
            best[2][50] += ["Powell"]
            best[2][5] += ["Cobyla"]
            best[48][208.333] += ["DiscreteDE"]
            best[48][20.8333] += ["pysot"]
            best[48][2.08333] += ["MultiCobyla"]
            best[48][0.208333] += ["NLOPT_LN_NELDERMEAD"]
            best[48][2083.33] += ["RPowell"]
            best[48][208.333] += ["CmaFmin2"]
            best[48][20.8333] += ["RPowell"]
            best[48][2.08333] += ["pysot"]
            best[48][0.208333] += ["BOBYQA"]
            best[48][20833.3] += ["MetaModelTwoPointsDE"]
            best[48][2083.33] += ["RPowell"]
            best[48][208.333] += ["DiscreteDE"]
            best[48][20.8333] += ["ChainNaiveTBPSACMAPowell"]
            best[48][2.08333] += ["NLOPT_LN_BOBYQA"]
            best[48][0.208333] += ["BOBYQA"]
            best[5][200000] += ["RescaledCMA"]
            best[5][20000] += ["RFMetaModelDE"]
            best[5][2000] += ["DiscreteDE"]
            best[5][200] += ["TwoPointsDE"]
            best[5][20] += ["NGOpt39"]
            best[5][2] += ["OnePlusLambda"]
            best[5][200000] += ["BAR"]
            best[5][20000] += ["ChainNaiveTBPSACMAPowell"]
            best[5][2000] += ["CmaFmin2"]
            best[5][200] += ["RotatedTwoPointsDE"]
            best[5][20] += ["pysot"]
            best[5][2] += ["NLOPT_LN_SBPLX"]
            best[5][200000] += ["ASCMADEthird"]
            best[5][20000] += ["ASCMADEthird"]
            best[5][2000] += ["QOTPDE"]
            best[5][200] += ["NGOpt10"]
            best[5][20] += ["LQODE"]
            best[5][2] += ["NLOPT_LN_SBPLX"]
            best[96][104.167] += ["NGOptF"]
            best[96][10.4167] += ["NGOpt4"]
            best[96][1.04167] += ["RPowell"]
            best[96][0.104167] += ["ASCMADEthird"]
            best[96][1041.67] += ["NLOPT_LN_NELDERMEAD"]
            best[96][104.167] += ["RPowell"]
            best[96][10.4167] += ["NGOpt8"]
            best[96][1.04167] += ["RPowell"]
            best[96][0.104167] += ["NLOPT_LN_NELDERMEAD"]
            best[96][1041.67] += ["CMandAS3"]
            best[96][104.167] += ["Powell"]
            best[96][10.4167] += ["NGOptBase"]
            best[96][1.04167] += ["Powell"]
            best[96][0.104167] += ["NLOPT_LN_NELDERMEAD"]
            bestdist = float("inf")
            for d1 in best:
                for nod2 in best[d1]:
                    dist = (d - d1) ** 2 + (nod - nod2) ** 2
                    if dist < bestdist:
                        bestdist = dist
                        bestalg = best[d1][nod2]
            return bestalg

        if self.fully_continuous and not self.has_noise:
            algs = recommend_method(self.dimension, self.budget / self.dimension)  # type: ignore
            if self.num_workers > 1:
                algs = [a for a in algs if not registry[a].no_parallelization]
                if len(algs) == 0:
                    return SQPCMA

            def most_frequent(List):
                return max(set(List), key=List.count)

            return registry[most_frequent(algs)]

        if self.fully_continuous and not self.has_noise and self.budget >= 12 * self.dimension:  # type: ignore
            return ConfPortfolio(optimizers=[GeneticDE, PSO, NGOpt39], warmup_ratio=0.33)
        else:
            return super()._select_optimizer_cls()


@registry.register
class NGOpt(NGOpt39):
    # Learning something automatically so that it's less unreadable would be great.
    pass


@registry.register
class Wiz(NGOpt16):
    def _select_optimizer_cls(self) -> base.OptCls:
        if self.fully_continuous and self.has_noise:  # In particular for neuro-DPS.
            DeterministicMix = ConfPortfolio(optimizers=[DiagonalCMA, PSO, GeneticDE])
            return Chaining([DeterministicMix, OptimisticNoisyOnePlusOne], ["half"])

        cma_vars = max(1, 4 + int(3 * np.log(self.dimension)))
        num36 = (
            1 + int(np.sqrt(4.0 * (4 * self.budget) // (self.dimension * 1000)))
            if self.budget is not None
            else 1
        )
        num21 = 1 + (4 * self.budget) // (self.dimension * 1000) if self.budget is not None else 1
        num_dim10 = (
            1 + int(np.sqrt(8.0 * (8 * self.budget) // (self.dimension * 1000)))
            if self.budget is not None
            else 1
        )
        num_dim20 = self.budget // (500 * self.dimension) if self.budget is not None else 1
        para = 1
        if self.budget is not None and self.budget > 5000 * self.dimension:
            para = num36 * cma_vars
        elif self.dimension < 5:
            para = num21 * cma_vars
        elif self.dimension < 10:
            para = num_dim10 * cma_vars
        elif self.dimension < 20:
            para = num_dim20 * cma_vars
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget < 1000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
        ):
            return Carola2
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget > 10000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
        ):
            return Carola2
        # Special cases in the bounded case
        if (
            self.budget is not None
            and self.budget > 500 * self.dimension
            and self.fully_continuous
            and not self.has_noise
            and self.num_objectives < 2
            and self.num_workers <= para
            and p.helpers.Normalizer(self.parametrization).fully_bounded
        ):
            if self.dimension == 1:
                return NGOpt16
            if (
                self.budget > 5000 * self.dimension
            ):  # Asymptotically let us trust NGOpt36 and its subtle restart.
                return NGOpt36
            if self.dimension < 5:  # Low dimension: let us hit the bounds.
                return NGOpt21
            if self.dimension < 10:  # Moderate dimension: reasonable restart + bet and run.
                num = 1 + int(np.sqrt(8.0 * (8 * self.budget) // (self.dimension * 1000)))
                return ConfPortfolio(optimizers=[NGOpt14] * num, warmup_ratio=0.7)
            if self.dimension < 20:  # Nobody knows why this seems to be so good.
                num = self.budget // (500 * self.dimension)
                return ConfPortfolio(
                    optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.3**i) for i in range(num)],
                    warmup_ratio=0.5,
                )
            if self.num_workers == 1:
                return CmaFmin2
            # We need a special case for dim < 30 ---> let's see later.
            # Otherwise, let us go back to normal life: NGOpt16 which rocks in many cases, possibly Cobyla.
            return NGOpt16
        elif (  # This might be specific of high-precision cases.
            self.budget is not None
            and self.fully_continuous
            and not self.has_noise
            and self.num_objectives < 2
            and self.num_workers <= cma_vars
            and self.budget > 50 * self.dimension
            and p.helpers.Normalizer(self.parametrization).fully_bounded
        ):
            if self.dimension < 3:
                return NGOpt8
            if self.dimension <= 20 and self.num_workers == 1:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            return NGOpt15
        else:
            return super()._select_optimizer_cls()


@registry.register
class NgIoh(NGOpt16):
    def _select_optimizer_cls(self) -> base.OptCls:
        if self.fully_continuous and self.has_noise:  # In particular for neuro-DPS.
            DeterministicMix = ConfPortfolio(optimizers=[DiagonalCMA, PSO, GeneticDE])
            return Chaining([DeterministicMix, OptimisticNoisyOnePlusOne], ["half"])

        cma_vars = max(1, 4 + int(3 * np.log(self.dimension)))
        num36 = (
            1 + int(np.sqrt(4.0 * (4 * self.budget) // (self.dimension * 1000)))
            if self.budget is not None
            else 1
        )
        num21 = 1 + (4 * self.budget) // (self.dimension * 1000) if self.budget is not None else 1
        num_dim10 = (
            1 + int(np.sqrt(8.0 * (8 * self.budget) // (self.dimension * 1000)))
            if self.budget is not None
            else 1
        )
        num_dim20 = self.budget // (500 * self.dimension) if self.budget is not None else 1
        para = 1
        if self.budget is not None and self.budget > 5000 * self.dimension:
            para = num36 * cma_vars
        elif self.dimension < 5:
            para = num21 * cma_vars
        elif self.dimension < 10:
            para = num_dim10 * cma_vars
        elif self.dimension < 20:
            para = num_dim20 * cma_vars
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget < 1000 * self.dimension
            and self.budget > 20 * self.dimension
            and not self.has_noise
            and self.dimension > 1
        ):
            return Carola2
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget > 10000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
        ):
            return Carola2
        # Special cases in the bounded case
        if (
            self.budget is not None
            and self.budget > 500 * self.dimension
            and self.fully_continuous
            and not self.has_noise
            and self.num_objectives < 2
            and self.num_workers <= para
            and p.helpers.Normalizer(self.parametrization).fully_bounded
        ):
            if self.dimension == 1:
                return NGOpt16
            if (
                self.budget > 5000 * self.dimension
            ):  # Asymptotically let us trust NGOpt36 and its subtle restart.
                return NGOpt36
            if self.dimension < 5:  # Low dimension: let us hit the bounds.
                return NGOpt21
            if self.dimension < 10:  # Moderate dimension: reasonable restart + bet and run.
                num = 1 + int(np.sqrt(8.0 * (8 * self.budget) // (self.dimension * 1000)))
                return ConfPortfolio(optimizers=[NGOpt14] * num, warmup_ratio=0.7)
            if self.dimension < 20:  # Nobody knows why this seems to be so good.
                num = self.budget // (500 * self.dimension)
                return ConfPortfolio(
                    optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.3**i) for i in range(num)],
                    warmup_ratio=0.5,
                )
            if self.num_workers == 1:
                return CmaFmin2
            # We need a special case for dim < 30 ---> let's see later.
            # Otherwise, let us go back to normal life: NGOpt16 which rocks in many cases, possibly Cobyla.
            return NGOpt16
        elif (  # This might be specific of high-precision cases.
            self.budget is not None
            and self.fully_continuous
            and not self.has_noise
            and self.num_objectives < 2
            and self.num_workers <= cma_vars
            and self.budget > 50 * self.dimension
            and p.helpers.Normalizer(self.parametrization).fully_bounded
        ):
            if self.dimension < 3:
                return NGOpt8
            if self.dimension <= 20 and self.num_workers == 1:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            return NGOpt15
        else:
            return super()._select_optimizer_cls()


@registry.register
class NgIoh2(NGOpt16):
    def _select_optimizer_cls(self) -> base.OptCls:
        if self.fully_continuous and self.has_noise:  # In particular for neuro-DPS.
            DeterministicMix = ConfPortfolio(optimizers=[DiagonalCMA, PSO, GeneticDE])
            return Chaining([DeterministicMix, OptimisticNoisyOnePlusOne], ["half"])

        cma_vars = max(1, 4 + int(3 * np.log(self.dimension)))
        num36 = (
            1 + int(np.sqrt(4.0 * (4 * self.budget) // (self.dimension * 1000)))
            if self.budget is not None
            else 1
        )
        num21 = 1 + (4 * self.budget) // (self.dimension * 1000) if self.budget is not None else 1
        num_dim10 = (
            1 + int(np.sqrt(8.0 * (8 * self.budget) // (self.dimension * 1000)))
            if self.budget is not None
            else 1
        )
        num_dim20 = self.budget // (500 * self.dimension) if self.budget is not None else 1
        para = 1
        if self.budget is not None and self.budget > 5000 * self.dimension:
            para = num36 * cma_vars
        elif self.dimension < 5:
            para = num21 * cma_vars
        elif self.dimension < 10:
            para = num_dim10 * cma_vars
        elif self.dimension < 20:
            para = num_dim20 * cma_vars
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget < 1000 * self.dimension
            and self.budget > 20 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 200
        ):
            return Carola2
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget >= 1000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 200
        ):
            return Carola2
        # Special cases in the bounded case
        if (
            self.budget is not None
            and self.budget > 500 * self.dimension
            and self.fully_continuous
            and not self.has_noise
            and self.num_objectives < 2
            and self.num_workers <= para
            and p.helpers.Normalizer(self.parametrization).fully_bounded
        ):
            if self.dimension == 1:
                return NGOpt16
            if (
                self.budget > 5000 * self.dimension
            ):  # Asymptotically let us trust NGOpt36 and its subtle restart.
                return NGOpt36
            if self.dimension < 5:  # Low dimension: let us hit the bounds.
                return NGOpt21
            if self.dimension < 10:  # Moderate dimension: reasonable restart + bet and run.
                num = 1 + int(np.sqrt(8.0 * (8 * self.budget) // (self.dimension * 1000)))
                return ConfPortfolio(optimizers=[NGOpt14] * num, warmup_ratio=0.7)
            if self.dimension < 20:  # Nobody knows why this seems to be so good.
                num = self.budget // (500 * self.dimension)
                return ConfPortfolio(
                    optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.3**i) for i in range(num)],
                    warmup_ratio=0.5,
                )
            if self.num_workers == 1:
                return CmaFmin2
            # We need a special case for dim < 30 ---> let's see later.
            # Otherwise, let us go back to normal life: NGOpt16 which rocks in many cases, possibly Cobyla.
            return NGOpt16
        elif (  # This might be specific of high-precision cases.
            self.budget is not None
            and self.fully_continuous
            and not self.has_noise
            and self.num_objectives < 2
            and self.num_workers <= cma_vars
            and self.budget > 50 * self.dimension
            and p.helpers.Normalizer(self.parametrization).fully_bounded
        ):
            if self.dimension < 3:
                return NGOpt8
            if self.dimension <= 20 and self.num_workers == 1:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            return NGOpt15
        else:
            return super()._select_optimizer_cls()


@registry.register
class NgIoh3(NGOpt16):
    def _select_optimizer_cls(self) -> base.OptCls:
        if self.fully_continuous and self.has_noise:  # In particular for neuro-DPS.
            DeterministicMix = ConfPortfolio(optimizers=[DiagonalCMA, PSO, GeneticDE])
            return Chaining([DeterministicMix, OptimisticNoisyOnePlusOne], ["half"])

        cma_vars = max(1, 4 + int(3 * np.log(self.dimension)))
        num36 = (
            1 + int(np.sqrt(4.0 * (4 * self.budget) // (self.dimension * 1000)))
            if self.budget is not None
            else 1
        )
        num21 = 1 + (4 * self.budget) // (self.dimension * 1000) if self.budget is not None else 1
        num_dim10 = (
            1 + int(np.sqrt(8.0 * (8 * self.budget) // (self.dimension * 1000)))
            if self.budget is not None
            else 1
        )
        num_dim20 = self.budget // (500 * self.dimension) if self.budget is not None else 1
        para = 1
        if self.budget is not None and self.budget > 5000 * self.dimension:
            para = num36 * cma_vars
        elif self.dimension < 5:
            para = num21 * cma_vars
        elif self.dimension < 10:
            para = num_dim10 * cma_vars
        elif self.dimension < 20:
            para = num_dim20 * cma_vars
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget < 1000 * self.dimension
            and self.budget > 20 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 100
        ):
            return Carola2
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget >= 1000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            return Carola2
        # Special cases in the bounded case
        if (
            self.budget is not None
            and self.budget > 500 * self.dimension
            and self.fully_continuous
            and not self.has_noise
            and self.num_objectives < 2
            and self.num_workers <= para
            and p.helpers.Normalizer(self.parametrization).fully_bounded
        ):
            if self.dimension == 1:
                return NGOpt16
            if (
                self.budget > 5000 * self.dimension
            ):  # Asymptotically let us trust NGOpt36 and its subtle restart.
                return NGOpt36
            if self.dimension < 5:  # Low dimension: let us hit the bounds.
                return NGOpt21
            if self.dimension < 10:  # Moderate dimension: reasonable restart + bet and run.
                num = 1 + int(np.sqrt(8.0 * (8 * self.budget) // (self.dimension * 1000)))
                return ConfPortfolio(optimizers=[NGOpt14] * num, warmup_ratio=0.7)
            if self.dimension < 20:  # Nobody knows why this seems to be so good.
                num = self.budget // (500 * self.dimension)
                return ConfPortfolio(
                    optimizers=[Rescaled(base_optimizer=NGOpt14, scale=1.3**i) for i in range(num)],
                    warmup_ratio=0.5,
                )
            if self.num_workers == 1:
                return CmaFmin2
            # We need a special case for dim < 30 ---> let's see later.
            # Otherwise, let us go back to normal life: NGOpt16 which rocks in many cases, possibly Cobyla.
            return NGOpt16
        elif (  # This might be specific of high-precision cases.
            self.budget is not None
            and self.fully_continuous
            and not self.has_noise
            and self.num_objectives < 2
            and self.num_workers <= cma_vars
            and self.budget > 50 * self.dimension
            and p.helpers.Normalizer(self.parametrization).fully_bounded
        ):
            if self.dimension < 3:
                return NGOpt8
            if self.dimension <= 20 and self.num_workers == 1:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            return NGOpt15
        else:
            return super()._select_optimizer_cls()


@registry.register
class NgIoh4(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self) -> base.OptCls:
        optCls: base.OptCls = NGOptBase
        funcinfo = self.parametrization.function
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget < 1000 * self.dimension
            and self.budget > 20 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 100
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget >= 1000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        # Special cases in the bounded case
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        elif self.dimension >= 60 and not funcinfo.metrizable:
            optCls = CMA
        # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, we choose {optCls}")
        return optCls


@registry.register
class NgIohRW2(NgIoh4):
    def _select_optimizer_cls(self) -> base.OptCls:
        if self.fully_continuous and not self.has_noise and self.budget >= 12 * self.dimension:  # type: ignore
            return ConfPortfolio(optimizers=[SQPCMA, QODE, PSO], warmup_ratio=0.33)
            # return ConfPortfolio(optimizers=[QODE, PSO, super()._select_optimizer_cls()], warmup_ratio=0.33)
        else:
            return super()._select_optimizer_cls()


@registry.register
class NgIoh5(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self) -> base.OptCls:
        optCls: base.OptCls = NGOptBase
        funcinfo = self.parametrization.function
        if not self.has_noise and self._arity > 0:
            return DiscreteLenglerOnePlusOne
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget < 1000 * self.dimension
            and self.budget > 20 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 100
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget >= 1000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        # Special cases in the bounded case
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        elif self.dimension >= 60 and not funcinfo.metrizable:
            optCls = CMA
        # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, we choose {optCls}")
        return optCls


@registry.register
class NgIoh6(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self) -> base.OptCls:
        optCls: base.OptCls = NGOptBase
        funcinfo = self.parametrization.function
        if not self.has_noise and self._arity > 0:
            return RecombiningPortfolioDiscreteOnePlusOne
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget < 1000 * self.dimension
            and self.budget > 20 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 100
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget >= 1000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        # Special cases in the bounded case
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        elif self.dimension >= 60 and not funcinfo.metrizable:
            optCls = CMA
        # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, we choose {optCls}")
        return optCls


class _MSR(Portfolio):
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
        super().__init__(
            parametrization,
            budget=budget,
            num_workers=num_workers,
            config=ConfPortfolio(optimizers=[base_optimizer] * num_single_runs),
        )
        self.coeffs: tp.List[np.ndarray] = []

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        if not self.coeffs:
            self.coeffs = [
                self.parametrization.random_state.uniform(size=self.num_objectives) for _ in self.optims
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


# Smooth variants of evolutionary algorithms.
SmoothDiscreteOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation="discrete").set_name(
    "SmoothDiscreteOnePlusOne", register=True
)
SmoothPortfolioDiscreteOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation="portfolio").set_name(
    "SmoothPortfolioDiscreteOnePlusOne", register=True
)
SmoothDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation="lengler").set_name(
    "SmoothDiscreteLenglerOnePlusOne", register=True
)
SmoothDiscreteLognormalOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation="lognormal").set_name(
    "SmoothDiscreteLognormalOnePlusOne", register=True
)
SuperSmoothDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(
    smoother=True, mutation="lengler", antismooth=9
).set_name("SuperSmoothDiscreteLenglerOnePlusOne", register=True)
UltraSmoothDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(
    smoother=True, mutation="lengler", antismooth=3
).set_name("UltraSmoothDiscreteLenglerOnePlusOne", register=True)
SmoothLognormalDiscreteOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation="lognormal").set_name(
    "SmoothLognormalDiscreteOnePlusOne", register=True
)
SmoothAdaptiveDiscreteOnePlusOne = ParametrizedOnePlusOne(smoother=True, mutation="adaptive").set_name(
    "SmoothAdaptiveDiscreteOnePlusOne", register=True
)

SmoothRecombiningPortfolioDiscreteOnePlusOne = ParametrizedOnePlusOne(
    smoother=True,
    crossover=True,
    mutation="portfolio",
).set_name("SmoothRecombiningPortfolioDiscreteOnePlusOne", register=True)
SmoothRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(
    smoother=True,
    crossover=True,
    mutation="lengler",
).set_name("SmoothRecombiningDiscreteLanglerOnePlusOne", register=True)
UltraSmoothRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(
    smoother=True,
    crossover=True,
    mutation="lengler",
    antismooth=3,
).set_name("UltraSmoothRecombiningDiscreteLanglerOnePlusOne", register=True)
UltraSmoothElitistRecombiningDiscreteLognormalOnePlusOne = ParametrizedOnePlusOne(
    smoother=True,
    crossover=True,
    mutation="lognormal",
    antismooth=3,
    roulette_size=7,
).set_name("UltraSmoothElitistRecombiningDiscreteLognormalOnePlusOne", register=True)
UltraSmoothElitistRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(
    smoother=True,
    crossover=True,
    mutation="lengler",
    antismooth=3,
    roulette_size=7,
).set_name("UltraSmoothElitistRecombiningDiscreteLanglerOnePlusOne", register=True)
SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(
    smoother=True,
    crossover=True,
    mutation="lengler",
    antismooth=9,
    roulette_size=7,
).set_name("SuperSmoothElitistRecombiningDiscreteLanglerOnePlusOne", register=True)
SuperSmoothRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(
    smoother=True,
    crossover=True,
    mutation="lengler",
    antismooth=9,
).set_name("SuperSmoothRecombiningDiscreteLanglerOnePlusOne", register=True)
SuperSmoothRecombiningDiscreteLognormalOnePlusOne = ParametrizedOnePlusOne(
    smoother=True,
    crossover=True,
    mutation="lognormal",
    antismooth=9,
).set_name("SuperSmoothRecombiningDiscreteLognormalOnePlusOne", register=True)
SmoothElitistRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(
    smoother=True,
    crossover=True,
    mutation="lengler",
    roulette_size=7,
).set_name("SmoothElitistRecombiningDiscreteLanglerOnePlusOne", register=True)
SmoothElitistRandRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(
    smoother=True,
    crossover=True,
    mutation="lengler",
    roulette_size=7,
    crossover_type="rand",
).set_name("SmoothElitistRandRecombiningDiscreteLanglerOnePlusOne", register=True)
SmoothElitistRandRecombiningDiscreteLognormalOnePlusOne = ParametrizedOnePlusOne(
    smoother=True,
    crossover=True,
    mutation="lognormal",
    roulette_size=7,
    crossover_type="rand",
).set_name("SmoothElitistRandRecombiningDiscreteLognormalOnePlusOne", register=True)
RecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(
    crossover=True,
    mutation="lengler",
).set_name("RecombiningDiscreteLanglerOnePlusOne", register=True)
RecombiningDiscreteLognormalOnePlusOne = ParametrizedOnePlusOne(
    crossover=True,
    mutation="lognormal",
).set_name("RecombiningDiscreteLognormalOnePlusOne", register=True)
MaxRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(
    crossover=True, mutation="lengler", crossover_type="max"
).set_name("MaxRecombiningDiscreteLenglerOnePlusOne", register=True)
MinRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(
    crossover=True, mutation="lengler", crossover_type="min"
).set_name("MinRecombiningDiscreteLenglerOnePlusOne", register=True)
OnePtRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(
    crossover=True, mutation="lengler", crossover_type="onepoint"
).set_name("OnePtRecombiningDiscreteLenglerOnePlusOne", register=True)
TwoPtRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(
    crossover=True, mutation="lengler", crossover_type="twopoint"
).set_name("TwoPtRecombiningDiscreteLenglerOnePlusOne", register=True)
RandRecombiningDiscreteLenglerOnePlusOne = ParametrizedOnePlusOne(
    crossover=True,
    mutation="lengler",
    crossover_type="rand",
).set_name("RandRecombiningDiscreteLenglerOnePlusOne", register=True)
RandRecombiningDiscreteLognormalOnePlusOne = ParametrizedOnePlusOne(
    crossover=True,
    mutation="lognormal",
    crossover_type="rand",
).set_name("RandRecombiningDiscreteLognormalOnePlusOne", register=True)


@registry.register
class NgIoh7(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self) -> base.OptCls:
        optCls: base.OptCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and not self.fully_continuous and not self.has_noise:
            return ConfPortfolio(
                optimizers=[
                    SuperSmoothDiscreteLenglerOnePlusOne,
                    SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne,
                    DiscreteLenglerOnePlusOne,
                ],
                warmup_ratio=0.4,
            )
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget < 1000 * self.dimension
            and self.budget > 20 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 100
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget >= 1000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        # Special cases in the bounded case
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        elif self.dimension >= 60 and not funcinfo.metrizable:
            optCls = CMA
        # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, we choose {optCls}")
        return optCls


@registry.register
class NgDS11(NGOptDSBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls:
        # print(f"NgIoh11 with dimension {self.dimension} and budget {budget}/{self.budget}")
        if budget is None:
            budget = self.budget
        else:
            self.budget = budget
        optCls: base.OptCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and not self.fully_continuous and not self.has_noise:
            return ConfPortfolio(
                optimizers=[
                    SuperSmoothDiscreteLenglerOnePlusOne,
                    SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne,
                    DiscreteLenglerOnePlusOne,
                ],
                warmup_ratio=0.4,
            )
        if self.fully_continuous and self.num_workers == 1 and budget is not None and not self.has_noise:
            if 300 * self.dimension < budget < 3000 * self.dimension:
                if self.dimension == 2:
                    return DS14
                if self.dimension < 4:
                    return DS4
                if self.dimension < 8:
                    return DS5
                if self.dimension < 15:
                    return DS9
                if self.dimension < 30:
                    return DS8
                if self.dimension < 60:
                    return DS9

            if 300 * self.dimension < budget < 3000 * self.dimension and self.dimension == 2:
                return DS6
            if 300 * self.dimension < budget < 3000 * self.dimension:
                return DS6
            if 3000 * self.dimension < budget:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return ChainMetaModelSQP
            if budget < 30 * self.dimension and self.dimension < 50 and self.dimension > 30:
                return ChainMetaModelSQP
            if budget >= 30 * self.dimension and budget < 300 * self.dimension and self.dimension == 2:
                return NLOPT_LN_SBPLX
            if budget >= 30 * self.dimension and budget < 300 * self.dimension and self.dimension < 15:
                return ChainMetaModelSQP
            if budget >= 300 * self.dimension and budget < 3000 * self.dimension and self.dimension < 30:
                return MultiCMA

        if (
            self.fully_continuous
            and self.num_workers == 1
            and budget is not None
            and budget < 1000 * self.dimension
            and budget > 20 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 100
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        if (
            self.fully_continuous
            and self.num_workers == 1
            and budget is not None
            and budget >= 1000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        # Special cases in the bounded case
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, we choose {optCls}")
        return optCls


@registry.register
class NgIoh11(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls:
        # print(f"NgIoh11 with dimension {self.dimension} and budget {budget}/{self.budget}")
        if budget is None:
            budget = self.budget
        else:
            self.budget = budget
        optCls: base.OptCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and not self.fully_continuous and not self.has_noise:
            return ConfPortfolio(
                optimizers=[
                    SuperSmoothDiscreteLenglerOnePlusOne,
                    SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne,
                    DiscreteLenglerOnePlusOne,
                ],
                warmup_ratio=0.4,
            )
        if self.fully_continuous and self.num_workers == 1 and budget is not None and not self.has_noise:
            if 300 * self.dimension < budget < 3000 * self.dimension:
                if self.dimension == 2:
                    return Carola14
                if self.dimension < 4:
                    return Carola4
                if self.dimension < 8:
                    return Carola5
                if self.dimension < 15:
                    return Carola9
                if self.dimension < 30:
                    return Carola8
                if self.dimension < 60:
                    return Carola9

            if 300 * self.dimension < budget < 3000 * self.dimension and self.dimension == 2:
                return FCarola6
            if 300 * self.dimension < budget < 3000 * self.dimension:
                return Carola6
            if 3000 * self.dimension < budget:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return ChainMetaModelSQP
            if budget < 30 * self.dimension and self.dimension < 50 and self.dimension > 30:
                return ChainMetaModelSQP
            if budget >= 30 * self.dimension and budget < 300 * self.dimension and self.dimension == 2:
                return NLOPT_LN_SBPLX
            if budget >= 30 * self.dimension and budget < 300 * self.dimension and self.dimension < 15:
                return ChainMetaModelSQP
            if budget >= 300 * self.dimension and budget < 3000 * self.dimension and self.dimension < 30:
                return MultiCMA

        if (
            self.fully_continuous
            and self.num_workers == 1
            and budget is not None
            and budget < 1000 * self.dimension
            and budget > 20 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 100
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        if (
            self.fully_continuous
            and self.num_workers == 1
            and budget is not None
            and budget >= 1000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        # Special cases in the bounded case
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, we choose {optCls}")
        return optCls


@registry.register
class NgIoh14(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls:
        assert budget is None
        optCls: base.OptCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and not self.fully_continuous and not self.has_noise:
            return ConfPortfolio(
                optimizers=[
                    SuperSmoothDiscreteLenglerOnePlusOne,
                    SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne,
                    DiscreteLenglerOnePlusOne,
                ],
                warmup_ratio=0.4,
            )
        if self.fully_continuous and self.budget is not None and not self.has_noise:
            num = self.budget // (1000 * self.dimension)
            if self.budget > 2000 * self.dimension and num >= self.num_workers:
                optimizers = []  # type: ignore
                # optimizers += [Rescaled(base_optimizer=NgIoh11, scale=max(.01, np.exp(-1.0 / np.random.rand())))]
                optimizers += [
                    Rescaled(
                        base_optimizer=NgIoh11._select_optimizer_cls(self),
                        scale=max(0.01, np.exp(-1.0 / np.random.rand())),
                    )
                ]
                if len(optimizers) < num:
                    optimizers += [FCarola6]  # type: ignore
                # if len(optimizers) < num:
                #    optimizers += [pCarola6]
                if len(optimizers) < num:
                    optimizers += [ChainMetaModelSQP]  # type: ignore
                if len(optimizers) < num:
                    MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                    optimizers += [MetaModelFmin2]  # type: ignore
                while len(optimizers) < num:
                    optimizers += [
                        Rescaled(
                            base_optimizer=NgIoh11._select_optimizer_cls(self),
                            scale=max(0.01, np.exp(-1.0 / np.random.rand())),
                        )
                    ]
                return ConfPortfolio(optimizers=optimizers, warmup_ratio=0.70, no_crossing=True)

        if self.fully_continuous and self.num_workers == 1 and self.budget is not None and not self.has_noise:
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                if self.dimension == 2:
                    return Carola14
                if self.dimension < 4:
                    return Carola4
                if self.dimension < 8:
                    return Carola5
                if self.dimension < 15:
                    return Carola9
                if self.dimension < 30:
                    return Carola8
                if self.dimension < 60:
                    return Carola9

            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension == 2:
                return FCarola6
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                return Carola6
            if 3000 * self.dimension < self.budget:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return ChainMetaModelSQP
            if self.budget < 30 * self.dimension and self.dimension < 50 and self.dimension > 30:
                return ChainMetaModelSQP
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension == 2
            ):
                return NLOPT_LN_SBPLX
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension < 15
            ):
                return ChainMetaModelSQP
            if (
                self.budget >= 300 * self.dimension
                and self.budget < 3000 * self.dimension
                and self.dimension < 30
            ):
                return MultiCMA

        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget < 1000 * self.dimension
            and self.budget > 20 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 100
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget >= 1000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        # Special cases in the bounded case
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, we choose {optCls}")
        return optCls


@registry.register
class NgIoh13(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls:
        assert budget is None
        optCls: base.OptCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and not self.fully_continuous and not self.has_noise:
            return ConfPortfolio(
                optimizers=[
                    SuperSmoothDiscreteLenglerOnePlusOne,
                    SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne,
                    DiscreteLenglerOnePlusOne,
                ],
                warmup_ratio=0.4,
            )
        if self.fully_continuous and self.budget is not None and not self.has_noise:
            num = self.budget // (1000 * self.dimension)
            if self.budget > 2000 * self.dimension and num >= self.num_workers:
                optimizers = []
                optimizers += [
                    Rescaled(
                        base_optimizer=NgIoh11._select_optimizer_cls(self),
                        scale=max(0.01, np.exp(-1.0 / np.random.rand())),
                    )
                ]
                if len(optimizers) < num:
                    optimizers += [FCarola6]  # type: ignore
                # if len(optimizers) < num:
                #    optimizers += [pCarola6]
                if len(optimizers) < num:
                    optimizers += [ChainMetaModelSQP]  # type: ignore
                if len(optimizers) < num:
                    MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                    optimizers += [MetaModelFmin2]  # type: ignore
                while len(optimizers) < num:
                    optimizers += [
                        Rescaled(
                            base_optimizer=NgIoh11._select_optimizer_cls(self),
                            scale=max(0.01, np.exp(-1.0 / np.random.rand())),
                        )
                    ]
                return ConfPortfolio(optimizers=optimizers, warmup_ratio=1.00, no_crossing=True)

        if self.fully_continuous and self.num_workers == 1 and self.budget is not None and not self.has_noise:
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                if self.dimension == 2:
                    return Carola14
                if self.dimension < 4:
                    return Carola4
                if self.dimension < 8:
                    return Carola5
                if self.dimension < 15:
                    return Carola9
                if self.dimension < 30:
                    return Carola8
                if self.dimension < 60:
                    return Carola9

            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension == 2:
                return FCarola6
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                return Carola6
            if 3000 * self.dimension < self.budget:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return ChainMetaModelSQP
            if self.budget < 30 * self.dimension and self.dimension < 50 and self.dimension > 30:
                return ChainMetaModelSQP
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension == 2
            ):
                return NLOPT_LN_SBPLX
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension < 15
            ):
                return ChainMetaModelSQP
            if (
                self.budget >= 300 * self.dimension
                and self.budget < 3000 * self.dimension
                and self.dimension < 30
            ):
                return MultiCMA

        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget < 1000 * self.dimension
            and self.budget > 20 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 100
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget >= 1000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        # Special cases in the bounded case
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, we choose {optCls}")
        return optCls


@registry.register
class NgIoh15(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls:
        assert budget is None
        optCls: base.OptCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and not self.fully_continuous and not self.has_noise:
            return ConfPortfolio(
                optimizers=[
                    SuperSmoothDiscreteLenglerOnePlusOne,
                    SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne,
                    DiscreteLenglerOnePlusOne,
                ],
                warmup_ratio=0.4,
            )
        if self.fully_continuous and self.budget is not None and not self.has_noise:
            num = self.budget // (1000 * self.dimension)
            if self.budget > 2000 * self.dimension and num >= self.num_workers:
                optimizers = []
                for _ in range(num):
                    optimizers += [
                        Rescaled(
                            base_optimizer=NgIoh11._select_optimizer_cls(self),
                            scale=max(0.01, np.exp(-1.0 / np.random.rand())),
                        )
                    ]
                    # optimizers += [NgIoh11]
                return ConfPortfolio(optimizers=optimizers, warmup_ratio=0.70, no_crossing=True)
        if self.fully_continuous and self.num_workers == 1 and self.budget is not None and not self.has_noise:
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                if self.dimension == 2:
                    return Carola14
                if self.dimension < 4:
                    return Carola4
                if self.dimension < 8:
                    return Carola5
                if self.dimension < 15:
                    return Carola9
                if self.dimension < 30:
                    return Carola8
                if self.dimension < 60:
                    return Carola9

            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension == 2:
                return FCarola6
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                return Carola6
            if 3000 * self.dimension < self.budget:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return ChainMetaModelSQP
            if self.budget < 30 * self.dimension and self.dimension < 50 and self.dimension > 30:
                return ChainMetaModelSQP
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension == 2
            ):
                return NLOPT_LN_SBPLX
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension < 15
            ):
                return ChainMetaModelSQP
            if (
                self.budget >= 300 * self.dimension
                and self.budget < 3000 * self.dimension
                and self.dimension < 30
            ):
                return MultiCMA

        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget < 1000 * self.dimension
            and self.budget > 20 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 100
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget >= 1000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        # Special cases in the bounded case
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, we choose {optCls}")
        return optCls


@registry.register
class NgIoh12(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls:
        assert budget is None
        optCls: base.OptCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and not self.fully_continuous and not self.has_noise:
            return ConfPortfolio(
                optimizers=[
                    SuperSmoothDiscreteLenglerOnePlusOne,
                    SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne,
                    DiscreteLenglerOnePlusOne,
                ],
                warmup_ratio=0.4,
            )
        if self.fully_continuous and self.budget is not None and not self.has_noise:
            num = self.budget // (1000 * self.dimension)
            if self.budget > 2000 * self.dimension and num >= self.num_workers:
                optimizers = []
                for _ in range(num):
                    optimizers += [
                        Rescaled(
                            base_optimizer=NgIoh11._select_optimizer_cls(self),
                            scale=max(0.01, np.exp(-1.0 / np.random.rand())),
                        )
                    ]
                    # optimizers += [NgIoh11]
                    # optimizers += [NgIoh11]
                return ConfPortfolio(optimizers=optimizers, warmup_ratio=1.00, no_crossing=True)
        if self.fully_continuous and self.num_workers == 1 and self.budget is not None and not self.has_noise:
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                if self.dimension == 2:
                    return Carola14
                if self.dimension < 4:
                    return Carola4
                if self.dimension < 8:
                    return Carola5
                if self.dimension < 15:
                    return Carola9
                if self.dimension < 30:
                    return Carola8
                if self.dimension < 60:
                    return Carola9

            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension == 2:
                return FCarola6
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                return Carola6
            if 3000 * self.dimension < self.budget:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return ChainMetaModelSQP
            if self.budget < 30 * self.dimension and self.dimension < 50 and self.dimension > 30:
                return ChainMetaModelSQP
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension == 2
            ):
                return NLOPT_LN_SBPLX
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension < 15
            ):
                return ChainMetaModelSQP
            if (
                self.budget >= 300 * self.dimension
                and self.budget < 3000 * self.dimension
                and self.dimension < 30
            ):
                return MultiCMA

        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget < 1000 * self.dimension
            and self.budget > 20 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 100
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget >= 1000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        # Special cases in the bounded case
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, we choose {optCls}")
        return optCls


@registry.register
class NgIoh16(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls:
        assert budget is None
        optCls: base.OptCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and not self.fully_continuous and not self.has_noise:
            return ConfPortfolio(
                optimizers=[
                    SuperSmoothDiscreteLenglerOnePlusOne,
                    SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne,
                    DiscreteLenglerOnePlusOne,
                ],
                warmup_ratio=0.4,
            )
        if self.fully_continuous and self.budget is not None and not self.has_noise:
            num = self.budget // (1000 * self.dimension)
            if self.budget > 2000 * self.dimension and num >= self.num_workers:
                optimizers = []
                # sub_budget = self.budget // num + (self.budget % num > 0)
                for _ in range(num):
                    optimizers += [
                        Rescaled(
                            base_optimizer=Carola14,
                            # base_optimizer=NgIoh11._select_optimizer_cls(self, sub_budget),
                            scale=max(0.01, np.exp(-1.0 / np.random.rand())),
                        )
                    ]
                # print("NgIoh16 chooses ", optimizers)
                return ConfPortfolio(optimizers=optimizers, warmup_ratio=1.00, no_crossing=True)
        if self.fully_continuous and self.num_workers == 1 and self.budget is not None and not self.has_noise:
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                if self.dimension == 2:
                    return Carola14
                if self.dimension < 4:
                    return Carola4
                if self.dimension < 8:
                    return Carola5
                if self.dimension < 15:
                    return Carola9
                if self.dimension < 30:
                    return Carola8
                if self.dimension < 60:
                    return Carola9

            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension == 2:
                return FCarola6
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                return Carola6
            if 3000 * self.dimension < self.budget:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return ChainMetaModelSQP
            if self.budget < 30 * self.dimension and self.dimension < 50 and self.dimension > 30:
                return ChainMetaModelSQP
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension == 2
            ):
                return NLOPT_LN_SBPLX
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension < 15
            ):
                return ChainMetaModelSQP
            if (
                self.budget >= 300 * self.dimension
                and self.budget < 3000 * self.dimension
                and self.dimension < 30
            ):
                return MultiCMA

        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget < 1000 * self.dimension
            and self.budget > 20 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 100
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget >= 1000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        # Special cases in the bounded case
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, we choose {optCls}")
        return optCls


@registry.register
class NgIoh17(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls:
        assert budget is None
        optCls: base.OptCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and not self.fully_continuous and not self.has_noise:
            return ConfPortfolio(
                optimizers=[
                    SuperSmoothDiscreteLenglerOnePlusOne,
                    SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne,
                    DiscreteLenglerOnePlusOne,
                ],
                warmup_ratio=0.4,
            )
        if self.fully_continuous and self.budget is not None and not self.has_noise:
            num = self.budget // (1000 * self.dimension)
            if self.budget > 2000 * self.dimension and num >= self.num_workers:
                optimizers = []
                orig_budget = self.budget
                sub_budget = self.budget // num + (self.budget % num > 0)
                # sub_budget = self.budget
                for _ in range(num):
                    optimizers += [
                        Rescaled(
                            # base_optimizer=Carola2,
                            base_optimizer=NgIoh11._select_optimizer_cls(self, sub_budget),
                            scale=max(0.01, np.exp(-1.0 / np.random.rand())),
                        )
                    ]
                    # optimizers += [NgIoh11]
                self.budget = orig_budget
                return Chaining(optimizers, [sub_budget] * (len(optimizers) - 1), no_crossing=True)
                # optimizers += [NgIoh11]
                # print("NgIoh16 chooses ", optimizers)
                # self.budget = orig_budget
                # return ConfPortfolio(optimizers=optimizers, warmup_ratio=1.00, no_crossing=True)
        if self.fully_continuous and self.num_workers == 1 and self.budget is not None and not self.has_noise:
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                if self.dimension == 2:
                    return Carola14
                if self.dimension < 4:
                    return Carola4
                if self.dimension < 8:
                    return Carola5
                if self.dimension < 15:
                    return Carola9
                if self.dimension < 30:
                    return Carola8
                if self.dimension < 60:
                    return Carola9

            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension == 2:
                return FCarola6
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                return Carola6
            if 3000 * self.dimension < self.budget:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return ChainMetaModelSQP
            if self.budget < 30 * self.dimension and self.dimension < 50 and self.dimension > 30:
                return ChainMetaModelSQP
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension == 2
            ):
                return NLOPT_LN_SBPLX
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension < 15
            ):
                return ChainMetaModelSQP
            if (
                self.budget >= 300 * self.dimension
                and self.budget < 3000 * self.dimension
                and self.dimension < 30
            ):
                return MultiCMA

        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget < 1000 * self.dimension
            and self.budget > 20 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 100
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget >= 1000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        # Special cases in the bounded case
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, we choose {optCls}")
        return optCls


@registry.register
class NgDS(NgDS11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls:
        assert budget is None
        optCls: base.OptCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and not self.fully_continuous and not self.has_noise:
            return ConfPortfolio(
                optimizers=[
                    SuperSmoothDiscreteLenglerOnePlusOne,
                    SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne,
                    DiscreteLenglerOnePlusOne,
                ],
                warmup_ratio=0.4,
            )
        if self.fully_continuous and self.budget is not None and not self.has_noise:
            if self.budget > 2000 * self.dimension:
                vlpcma = ParametrizedMetaModel(multivariate_optimizer=VLPCMA)
                return vlpcma
                # optimizers += [NgIoh11]
                # print("NgIoh16 chooses ", optimizers)
                # self.budget = orig_budget
                # return ConfPortfolio(optimizers=optimizers, warmup_ratio=1.00, no_crossing=True)
        if self.fully_continuous and self.num_workers == 1 and self.budget is not None and not self.has_noise:
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                if self.dimension == 2:
                    return DS14
                if self.dimension < 4:
                    return DS4
                if self.dimension < 8:
                    return DS5
                if self.dimension < 15:
                    return DS9
                if self.dimension < 30:
                    return DS8
                if self.dimension < 60:
                    return DS9

            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension == 2:
                return DS6
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                return DS6
            if 3000 * self.dimension < self.budget:
                MetaModelDS = ParametrizedMetaModel(multivariate_optimizer=DSproba)
                MetaModelDS.no_parallelization = True
                return MetaModelDS
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                return ChainMetaModelDSSQP
            if self.budget < 30 * self.dimension and self.dimension < 50 and self.dimension > 30:
                return ChainMetaModelDSSQP
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension == 2
            ):
                return NLOPT_LN_SBPLX
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension < 15
            ):
                return ChainMetaModelDSSQP
            if (
                self.budget >= 300 * self.dimension
                and self.budget < 3000 * self.dimension
                and self.dimension < 30
            ):
                return MultiDS  # TODOICI

        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget < 1000 * self.dimension
            and self.budget > 20 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 100
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return DS2
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget >= 1000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return DS2  # TODOICI
        # Special cases in the bounded case
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, we choose {optCls}")
        return optCls


@registry.register
class NgIoh21(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls:
        assert budget is None
        optCls: base.OptCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and not self.fully_continuous and not self.has_noise:
            return ConfPortfolio(
                optimizers=[
                    SuperSmoothDiscreteLenglerOnePlusOne,
                    SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne,
                    DiscreteLenglerOnePlusOne,
                ],
                warmup_ratio=0.4,
            )
        if self.fully_continuous and self.budget is not None and not self.has_noise:
            if self.budget > 2000 * self.dimension:
                vlpcma = ParametrizedMetaModel(multivariate_optimizer=VLPCMA)
                return vlpcma
                # optimizers += [NgIoh11]
                # print("NgIoh16 chooses ", optimizers)
                # self.budget = orig_budget
                # return ConfPortfolio(optimizers=optimizers, warmup_ratio=1.00, no_crossing=True)
        if self.fully_continuous and self.num_workers == 1 and self.budget is not None and not self.has_noise:
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                if self.dimension == 2:
                    return Carola14
                if self.dimension < 4:
                    return Carola4
                if self.dimension < 8:
                    return Carola5
                if self.dimension < 15:
                    return Carola9
                if self.dimension < 30:
                    return Carola8
                if self.dimension < 60:
                    return Carola9

            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension == 2:
                return FCarola6
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                return Carola6
            if 3000 * self.dimension < self.budget:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return ChainMetaModelSQP
            if self.budget < 30 * self.dimension and self.dimension < 50 and self.dimension > 30:
                return ChainMetaModelSQP
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension == 2
            ):
                return NLOPT_LN_SBPLX
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension < 15
            ):
                return ChainMetaModelSQP
            if (
                self.budget >= 300 * self.dimension
                and self.budget < 3000 * self.dimension
                and self.dimension < 30
            ):
                return MultiCMA

        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget < 1000 * self.dimension
            and self.budget > 20 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 100
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget >= 1000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        # Special cases in the bounded case
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, we choose {optCls}")
        return optCls


@registry.register
class NgDS2(NgDS11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls:
        if self.budget is not None and self.dimension < self.budget:
            return NgIoh21._select_optimizer_cls(self, budget)  # type: ignore
        assert budget is None
        optCls: base.OptCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and not self.fully_continuous and not self.has_noise:
            return ConfPortfolio(
                optimizers=[
                    SuperSmoothDiscreteLenglerOnePlusOne,
                    SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne,
                    DiscreteLenglerOnePlusOne,
                ],
                warmup_ratio=0.4,
            )
        if self.fully_continuous and self.budget is not None and not self.has_noise:
            if self.budget > 2000 * self.dimension:
                vlpcma = ParametrizedMetaModel(multivariate_optimizer=VLPCMA)
                return vlpcma
                # optimizers += [NgIoh11]
                # print("NgIoh16 chooses ", optimizers)
                # self.budget = orig_budget
                # return ConfPortfolio(optimizers=optimizers, warmup_ratio=1.00, no_crossing=True)
        if self.fully_continuous and self.num_workers == 1 and self.budget is not None and not self.has_noise:
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                if self.dimension == 2:
                    return DS14
                if self.dimension < 4:
                    return DS4
                if self.dimension < 8:
                    return DS5
                if self.dimension < 15:
                    return DS9
                if self.dimension < 30:
                    return DS8
                if self.dimension < 60:
                    return DS9

            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension == 2:
                return DS6
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                return DS6
            if 3000 * self.dimension < self.budget:
                MetaModelDS = ParametrizedMetaModel(multivariate_optimizer=DSproba)
                MetaModelDS.no_parallelization = True
                return MetaModelDS
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                return ChainMetaModelDSSQP
            if self.budget < 30 * self.dimension and self.dimension < 50 and self.dimension > 30:
                return ChainMetaModelDSSQP
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension == 2
            ):
                return NLOPT_LN_SBPLX
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension < 15
            ):
                return ChainMetaModelDSSQP
            if (
                self.budget >= 300 * self.dimension
                and self.budget < 3000 * self.dimension
                and self.dimension < 30
            ):
                return MultiDS  # TODOICI

        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget < 1000 * self.dimension
            and self.budget > 20 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 100
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return DS2
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget >= 1000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return DS2  # TODOICI
        # Special cases in the bounded case
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, we choose {optCls}")
        return optCls


@registry.register
class NGDSRW(NGOpt39):
    def _select_optimizer_cls(self) -> base.OptCls:
        if self.fully_continuous and not self.has_noise and self.budget >= 12 * self.dimension:  # type: ignore
            return ConfPortfolio(
                optimizers=[GeneticDE, PSO, super()._select_optimizer_cls()], warmup_ratio=0.33
            )
        else:
            if self.budget is not None and self.dimension > self.budget:
                return NgDS2._select_optimizer_cls(self)  # type: ignore
            return NGOpt39._select_optimizer_cls(self)  # type: ignore


@registry.register
class NgIoh20(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls:
        assert budget is None
        optCls: base.OptCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and not self.fully_continuous and not self.has_noise:
            return ConfPortfolio(
                optimizers=[
                    SuperSmoothDiscreteLenglerOnePlusOne,
                    SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne,
                    DiscreteLenglerOnePlusOne,
                ],
                warmup_ratio=0.4,
            )
        if self.fully_continuous and self.budget is not None and not self.has_noise:
            if self.budget > 2000 * self.dimension:
                vlpcma = (
                    ParametrizedMetaModel(multivariate_optimizer=VLPCMA)
                    if self.dimension > 4
                    else ParametrizedMetaModel(multivariate_optimizer=LPCMA)
                )
                return vlpcma
                # optimizers += [NgIoh11]
                # print("NgIoh16 chooses ", optimizers)
                # self.budget = orig_budget
                # return ConfPortfolio(optimizers=optimizers, warmup_ratio=1.00, no_crossing=True)
        if self.fully_continuous and self.num_workers == 1 and self.budget is not None and not self.has_noise:
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                if self.dimension == 2:
                    return Carola14
                if self.dimension < 4:
                    return Carola4
                if self.dimension < 8:
                    return Carola5
                if self.dimension < 15:
                    return Carola9
                if self.dimension < 30:
                    return Carola8
                if self.dimension < 60:
                    return Carola9

            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension == 2:
                return FCarola6
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                return Carola6
            if 3000 * self.dimension < self.budget:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return ChainMetaModelSQP
            if self.budget < 30 * self.dimension and self.dimension < 50 and self.dimension > 30:
                return ChainMetaModelSQP
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension == 2
            ):
                return NLOPT_LN_SBPLX
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension < 15
            ):
                return ChainMetaModelSQP
            if (
                self.budget >= 300 * self.dimension
                and self.budget < 3000 * self.dimension
                and self.dimension < 30
            ):
                return MultiCMA

        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget < 1000 * self.dimension
            and self.budget > 20 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 100
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget >= 1000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        # Special cases in the bounded case
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, we choose {optCls}")
        return optCls


@registry.register
class NgIoh19(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls:
        assert budget is None
        optCls: base.OptCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and not self.fully_continuous and not self.has_noise:
            return ConfPortfolio(
                optimizers=[
                    SuperSmoothDiscreteLenglerOnePlusOne,
                    SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne,
                    DiscreteLenglerOnePlusOne,
                ],
                warmup_ratio=0.4,
            )
        if self.fully_continuous and self.budget is not None and not self.has_noise:
            if self.budget > 2000 * self.dimension:
                vlpcma = VLPCMA if self.dimension > 4 else LPCMA
                return vlpcma
                # optimizers += [NgIoh11]
                # print("NgIoh16 chooses ", optimizers)
                # self.budget = orig_budget
                # return ConfPortfolio(optimizers=optimizers, warmup_ratio=1.00, no_crossing=True)
        if self.fully_continuous and self.num_workers == 1 and self.budget is not None and not self.has_noise:
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                if self.dimension == 2:
                    return Carola14
                if self.dimension < 4:
                    return Carola4
                if self.dimension < 8:
                    return Carola5
                if self.dimension < 15:
                    return Carola9
                if self.dimension < 30:
                    return Carola8
                if self.dimension < 60:
                    return Carola9

            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension == 2:
                return FCarola6
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                return Carola6
            if 3000 * self.dimension < self.budget:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return ChainMetaModelSQP
            if self.budget < 30 * self.dimension and self.dimension < 50 and self.dimension > 30:
                return ChainMetaModelSQP
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension == 2
            ):
                return NLOPT_LN_SBPLX
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension < 15
            ):
                return ChainMetaModelSQP
            if (
                self.budget >= 300 * self.dimension
                and self.budget < 3000 * self.dimension
                and self.dimension < 30
            ):
                return MultiCMA

        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget < 1000 * self.dimension
            and self.budget > 20 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 100
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget >= 1000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        # Special cases in the bounded case
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, we choose {optCls}")
        return optCls


@registry.register
class NgIoh18(NgIoh11):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls:
        assert budget is None
        optCls: base.OptCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and not self.fully_continuous and not self.has_noise:
            return ConfPortfolio(
                optimizers=[
                    SuperSmoothDiscreteLenglerOnePlusOne,
                    SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne,
                    DiscreteLenglerOnePlusOne,
                ],
                warmup_ratio=0.4,
            )
        if self.fully_continuous and self.budget is not None and not self.has_noise:
            num = self.budget // (1000 * self.dimension)
            if self.budget > 2000 * self.dimension and num >= self.num_workers:
                optimizers = []
                orig_budget = self.budget
                sub_budget = self.budget // num + (self.budget % num > 0)
                # sub_budget = self.budget
                optimizers = [NgIoh11._select_optimizer_cls(self, sub_budget)]
                for k in range(num - 1):
                    optimizers += (
                        [
                            Rescaled(
                                # base_optimizer=Carola2,
                                base_optimizer=NgIoh11._select_optimizer_cls(self, sub_budget),
                                scale=np.random.rand() * 2.0,
                            )
                        ]
                        if k % 3 == 2
                        else (
                            [
                                Rescaled(
                                    # base_optimizer=Carola2,
                                    base_optimizer=NgIoh11._select_optimizer_cls(self, sub_budget),
                                    shift=np.random.randn(),
                                )
                            ]
                            if k % 3 == 0
                            else [
                                Rescaled(
                                    # base_optimizer=Carola2,
                                    base_optimizer=NgIoh11._select_optimizer_cls(self, sub_budget),
                                    scale=np.random.rand() * 2.0,
                                    shift=np.random.randn(),
                                )
                            ]
                        )
                    )
                    # optimizers += [NgIoh11]
                self.budget = orig_budget
                return Chaining(optimizers, [sub_budget] * (len(optimizers) - 1), no_crossing=True)
                # optimizers += [NgIoh11]
                # print("NgIoh16 chooses ", optimizers)
                # self.budget = orig_budget
                # return ConfPortfolio(optimizers=optimizers, warmup_ratio=1.00, no_crossing=True)
        if self.fully_continuous and self.num_workers == 1 and self.budget is not None and not self.has_noise:
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                if self.dimension == 2:
                    return Carola14
                if self.dimension < 4:
                    return Carola4
                if self.dimension < 8:
                    return Carola5
                if self.dimension < 15:
                    return Carola9
                if self.dimension < 30:
                    return Carola8
                if self.dimension < 60:
                    return Carola9

            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension == 2:
                return FCarola6
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                return Carola6
            if 3000 * self.dimension < self.budget:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return ChainMetaModelSQP
            if self.budget < 30 * self.dimension and self.dimension < 50 and self.dimension > 30:
                return ChainMetaModelSQP
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension == 2
            ):
                return NLOPT_LN_SBPLX
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension < 15
            ):
                return ChainMetaModelSQP
            if (
                self.budget >= 300 * self.dimension
                and self.budget < 3000 * self.dimension
                and self.dimension < 30
            ):
                return MultiCMA

        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget < 1000 * self.dimension
            and self.budget > 20 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 100
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget >= 1000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        # Special cases in the bounded case
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, we choose {optCls}")
        return optCls


@registry.register
class NgIoh10(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self) -> base.OptCls:
        optCls: base.OptCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and not self.fully_continuous and not self.has_noise:
            return ConfPortfolio(
                optimizers=[
                    SuperSmoothDiscreteLenglerOnePlusOne,
                    SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne,
                    DiscreteLenglerOnePlusOne,
                ],
                warmup_ratio=0.4,
            )
        if self.fully_continuous and self.num_workers == 1 and self.budget is not None and not self.has_noise:
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension == 2:
                return FCarola6
            if 300 * self.dimension < self.budget < 3000 * self.dimension:
                return Carola6
            if 3000 * self.dimension < self.budget:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return ChainMetaModelSQP
            if self.budget < 30 * self.dimension and self.dimension < 50 and self.dimension > 30:
                return ChainMetaModelSQP
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension == 2
            ):
                return NLOPT_LN_SBPLX
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension < 15
            ):
                return ChainMetaModelSQP
            if (
                self.budget >= 300 * self.dimension
                and self.budget < 3000 * self.dimension
                and self.dimension < 30
            ):
                return MultiCMA

        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget < 1000 * self.dimension
            and self.budget > 20 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 100
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget >= 1000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        # Special cases in the bounded case
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, we choose {optCls}")
        return optCls


@registry.register
class NgIoh9(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self) -> base.OptCls:
        optCls: base.OptCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and not self.fully_continuous and not self.has_noise:
            return ConfPortfolio(
                optimizers=[
                    SuperSmoothDiscreteLenglerOnePlusOne,
                    SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne,
                    DiscreteLenglerOnePlusOne,
                ],
                warmup_ratio=0.4,
            )
        if self.fully_continuous and self.num_workers == 1 and self.budget is not None and not self.has_noise:
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension == 2:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget and self.dimension == 3:
                return ChainMetaModelSQP
            if 3000 * self.dimension < self.budget:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return MetaModelFmin2
            if 300 * self.dimension < self.budget < 3000 * self.dimension and self.dimension <= 3:
                MetaModelFmin2 = ParametrizedMetaModel(multivariate_optimizer=CmaFmin2)
                MetaModelFmin2.no_parallelization = True
                return ChainMetaModelSQP
            if self.budget < 30 * self.dimension and self.dimension < 50 and self.dimension > 30:
                return ChainMetaModelSQP
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension == 2
            ):
                return NLOPT_LN_SBPLX
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension < 15
            ):
                return ChainMetaModelSQP
            if (
                self.budget >= 300 * self.dimension
                and self.budget < 3000 * self.dimension
                and self.dimension < 30
            ):
                return MultiCMA

        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget < 1000 * self.dimension
            and self.budget > 20 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 100
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget >= 1000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        # Special cases in the bounded case
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, we choose {optCls}")
        return optCls


@registry.register
class NgIoh8(NGOptBase):
    """Nevergrad optimizer by competence map. You might modify this one for designing your own competence map."""

    def _select_optimizer_cls(self) -> base.OptCls:
        optCls: base.OptCls = NGOptBase
        funcinfo = self.parametrization.function
        if isinstance(self.parametrization, p.Array) and not self.fully_continuous and not self.has_noise:
            return ConfPortfolio(
                optimizers=[
                    SuperSmoothDiscreteLenglerOnePlusOne,
                    SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne,
                    DiscreteLenglerOnePlusOne,
                ],
                warmup_ratio=0.4,
            )
        if self.fully_continuous and self.num_workers == 1 and self.budget is not None and not self.has_noise:
            if self.budget < 30 * self.dimension and self.dimension < 50 and self.dimension > 30:
                return ChainMetaModelSQP
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension == 2
            ):
                return NLOPT_LN_SBPLX
            if (
                self.budget >= 30 * self.dimension
                and self.budget < 300 * self.dimension
                and self.dimension < 15
            ):
                return ChainMetaModelSQP
            if (
                self.budget >= 300 * self.dimension
                and self.budget < 3000 * self.dimension
                and self.dimension < 30
            ):
                return MultiCMA
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget < 1000 * self.dimension
            and self.budget > 20 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 100
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget >= 1000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return Carola2
        # Special cases in the bounded case
        if self.has_noise and (self.has_discrete_not_softmax or not funcinfo.metrizable):
            optCls = RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne
        elif self.dimension >= 60 and not funcinfo.metrizable:
            optCls = CMA
        # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, we choose {optCls}")
        return optCls


# Specifically for RL.
MixDeterministicRL = ConfPortfolio(optimizers=[DiagonalCMA, PSO, GeneticDE]).set_name(
    "MixDeterministicRL", register=True
)
SpecialRL = Chaining([MixDeterministicRL, TBPSA], ["half"]).set_name("SpecialRL", register=True)
NoisyRL1 = Chaining([MixDeterministicRL, NoisyOnePlusOne], ["half"]).set_name("NoisyRL1", register=True)
NoisyRL2 = Chaining(
    [MixDeterministicRL, RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne], ["half"]
).set_name("NoisyRL2", register=True)
NoisyRL3 = Chaining([MixDeterministicRL, OptimisticNoisyOnePlusOne], ["half"]).set_name(
    "NoisyRL3", register=True
)


from . import experimentalvariants  # pylint: disable=unused-import

FCarola6 = Chaining([NGOpt, NGOpt, RBFGS], ["tenth", "most"]).set_name("FCarola6", register=True)
FCarola6.no_parallelization = True
Carola11 = Chaining([MultiCMA, RBFGS], ["most"]).set_name("Carola11", register=True)
Carola11.no_parallelization = True
Carola14 = Chaining([MultiCMA, RBFGS], ["most"]).set_name("Carola14", register=True)
Carola14.no_parallelization = True
DS14 = Chaining([MultiDS, RBFGS], ["most"]).set_name("DS14", register=True)
DS14.no_parallelization = True
Carola13 = Chaining([CmaFmin2, RBFGS], ["most"]).set_name("Carola13", register=True)
Carola13.no_parallelization = True
Carola15 = Chaining([Cobyla, MetaModel, RBFGS], ["sqrt", "most"]).set_name("Carola15", register=True)
Carola15.no_parallelization = True


@registry.register
class NgIoh12b(NgIoh12):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.no_crossing = True


@registry.register
class NgIoh13b(NgIoh13):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.no_crossing = True


@registry.register
class NgIoh14b(NgIoh14):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.no_crossing = True


@registry.register
class NgIoh15b(NgIoh15):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.no_crossing = True


NgDS3 = Chaining([LognormalDiscreteOnePlusOne, NgDS2], ["tenth"]).set_name("NgDS3", register=True)
NgLn = Chaining([LognormalDiscreteOnePlusOne, NGOpt], ["tenth"]).set_name("NgLn", register=True)
NgRS = Chaining([oneshot.RandomSearch, NGOpt], ["tenth"]).set_name("NgRS", register=True)


@registry.register
class CSEC(NGOpt39):
    # Cognitive Synthetic Entity Controller

    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls:
        assert budget is None

        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget >= 3000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return NgIoh21._select_optimizer_cls(self, budget)  # type: ignore
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget >= 30 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return NgIoh4._select_optimizer_cls(self)  # type: ignore

        if (
            self.fully_continuous
            and self.budget is not None
            and self.num_workers > np.sqrt(self.budget)
            and self.budget >= 30 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return NgDS3

        return NgDS2._select_optimizer_cls(self, budget)  # type: ignore


@registry.register
class CSEC10(NGOptBase):
    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls:
        assert self.budget is not None
        function = self.parametrization
        # assert False
        if isinstance(self.parametrization, p.Array) and not self.fully_continuous and not self.has_noise:
            return GeneticDE
            # return ConfPortfolio(
            #    optimizers=[
            #        SuperSmoothDiscreteLenglerOnePlusOne,
            #        SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne,
            #        DiscreteLenglerOnePlusOne,
            #    ],
            #    warmup_ratio=0.4,
            # )
        if (
            function.real_world
            and not function.hptuning
            and not function.neural
            and not self.parametrization.function.metrizable
        ):
            return NgDS2._select_optimizer_cls(self, budget)  # type: ignore
        if function.has_constraints:
            return NgLn
        if (
            self.num_workers == 1
            and function.real_world
            and not function.hptuning
            and not function.neural
            and self.dimension > self.budget
            and self.fully_continuous
            and not self.has_noise
        ):
            return DSproba
        if (
            self.num_workers < np.sqrt(1 + self.dimension)
            and function.real_world
            and not function.hptuning
            and not function.neural
            and 8 * self.dimension > self.budget
            and self.fully_continuous
            and not self.has_noise
        ):
            return DiscreteLenglerOnePlusOne
        if function.real_world and not function.hptuning and not function.neural and self.fully_continuous:
            return NGOpt._select_optimizer_cls(self)  # type: ignore
        if (
            function.real_world
            and function.neural
            and not function.function.deterministic
            and not function.enforce_determinism
        ):
            return NoisyRL2
        if (
            function.real_world
            and function.neural
            and (function.function.deterministic or function.enforce_determinism)
        ):
            return SQOPSO
        if function.real_world and not function.neural:
            return NGDSRW._select_optimizer_cls(self)  # type: ignore
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget >= 3000 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return NgIoh21._select_optimizer_cls(self, budget)  # type: ignore
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget >= 30 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return NgIoh4._select_optimizer_cls(self)  # type: ignore

        if (
            self.fully_continuous
            and self.budget is not None
            and self.num_workers > np.log(3 + self.budget)
            and self.budget >= 30 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return NgDS3

        return NgDS2._select_optimizer_cls(self, budget)  # type: ignore


@registry.register
class CSEC11(NGOptBase):
    def _select_optimizer_cls(self, budget: tp.Optional[int] = None) -> base.OptCls:
        assert self.budget is not None
        function = self.parametrization
        # assert False
        if isinstance(self.parametrization, p.Array) and not self.fully_continuous and not self.has_noise:
            return GeneticDE
            # return ConfPortfolio(
            #    optimizers=[
            #        SuperSmoothDiscreteLenglerOnePlusOne,
            #        SuperSmoothElitistRecombiningDiscreteLenglerOnePlusOne,
            #        DiscreteLenglerOnePlusOne,
            #    ],
            #    warmup_ratio=0.4,
            # )
        if (
            function.real_world
            and not function.hptuning
            and not function.neural
            and not self.parametrization.function.metrizable
        ):
            return NgDS2._select_optimizer_cls(self, budget)  # type: ignore
        if function.has_constraints:
            return NgLn
        if (
            self.num_workers == 1
            and function.real_world
            and not function.hptuning
            and not function.neural
            and self.dimension > self.budget
            and self.fully_continuous
            and not self.has_noise
        ):
            return DSproba
        if (
            self.num_workers < np.sqrt(1 + self.dimension)
            and function.real_world
            and not function.hptuning
            and not function.neural
            and 8 * self.dimension > self.budget
            and self.fully_continuous
            and not self.has_noise
        ):
            return DiscreteLenglerOnePlusOne
        if function.real_world and not function.hptuning and not function.neural and self.fully_continuous:
            return NGOpt._select_optimizer_cls(self)  # type: ignore
        if (
            function.real_world
            and function.neural
            and not function.function.deterministic
            and not function.enforce_determinism
        ):
            return NoisyRL2
        if (
            function.real_world
            and function.neural
            and (function.function.deterministic or function.enforce_determinism)
        ):
            return SQOPSO
        if function.real_world and not function.neural:
            return NGDSRW._select_optimizer_cls(self)  # type: ignore
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget >= 300 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return NgIoh21._select_optimizer_cls(self, budget)  # type: ignore
        if (
            self.fully_continuous
            and self.num_workers == 1
            and self.budget is not None
            and self.budget >= 30 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return NgIoh4._select_optimizer_cls(self)  # type: ignore

        if (
            self.fully_continuous
            and self.budget is not None
            and self.num_workers > np.log(3 + self.budget)
            and self.budget >= 30 * self.dimension
            and not self.has_noise
            and self.dimension > 1
            and self.dimension < 50
        ):
            # print(f"budget={self.budget}, dim={self.dimension}, nw={self.num_workers}, Carola2")
            return NgDS3

        return NgDS2._select_optimizer_cls(self, budget)  # type: ignore


@registry.register
class NgIohTuned(CSEC11):
    # Learning something automatically so that it's less unreadable would be great.
    pass


# ExtLognormalDiscreteOnePlusOne = Chaining(
#    [LognormalDiscreteOnePlusOne] * 10, [0.1] * 9, no_crossing=True
# ).set_name("ExtLognormalDiscreteOnePlusOne", register=True)


SplitCSEC11 = ConfSplitOptimizer(
    multivariate_optimizer=CSEC11, monovariate_optimizer=CSEC11, non_deterministic_descriptor=False
).set_name("SplitCSEC11", register=True)
SplitSQOPSO = ConfSplitOptimizer(
    multivariate_optimizer=SQOPSO, monovariate_optimizer=SQOPSO, non_deterministic_descriptor=False
).set_name("SplitSQOPSO", register=True)
SplitPSO = ConfSplitOptimizer(
    multivariate_optimizer=PSO, monovariate_optimizer=PSO, non_deterministic_descriptor=False
).set_name("SplitPSO", register=True)
SplitCMA = ConfSplitOptimizer(
    multivariate_optimizer=CMA, monovariate_optimizer=CMA, non_deterministic_descriptor=False
).set_name("SplitCMA", register=True)
SplitQODE = ConfSplitOptimizer(
    multivariate_optimizer=QODE, monovariate_optimizer=QODE, non_deterministic_descriptor=False
).set_name("SplitQODE", register=True)
SplitTwoPointsDE = ConfSplitOptimizer(
    multivariate_optimizer=TwoPointsDE, monovariate_optimizer=TwoPointsDE, non_deterministic_descriptor=False
).set_name("SplitTwoPointsDE", register=True)
SplitDE = ConfSplitOptimizer(
    multivariate_optimizer=DE, monovariate_optimizer=DE, non_deterministic_descriptor=False
).set_name("SplitDE", register=True)


SQOPSODCMA = Chaining([SQOPSO, DiagonalCMA], ["half"]).set_name("SQOPSODCMA", register=True)
SQOPSODCMA20 = Chaining(optimizers=[SQOPSODCMA] * 20, budgets=["equal"] * 19, no_crossing=True).set_name(
    "SQOPSODCMA20", register=True
)
SQOPSODCMA20bar = ConfPortfolio(optimizers=[SQOPSODCMA] * 20, warmup_ratio=0.5).set_name(
    "SQOPSODCMA20bar", register=True
)
NgIohLn = Chaining([LognormalDiscreteOnePlusOne, CSEC11], ["tenth"]).set_name("NgIohLn", register=True)
NgIohRS = Chaining([oneshot.RandomSearch, CSEC11], ["tenth"]).set_name("NgIohRS", register=True)
MultiLN = ConfPortfolio(
    optimizers=[
        Rescaled(base_optimizer=SmallLognormalDiscreteOnePlusOne, scale=2.0 ** (i - 5)) for i in range(6)
    ],
    warmup_ratio=0.5,
).set_name("MultiLN", register=True)
NgIohMLn = Chaining([MultiLN, CSEC11], ["tenth"]).set_name("NgIohMLn", register=True)
