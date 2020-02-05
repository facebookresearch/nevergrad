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
from nevergrad.functions import MultiobjectiveFunction
from . import utils
from . import base
from . import mutations
from .base import registry as registry
from .base import addCompare
from .base import InefficientSettingsWarning as InefficientSettingsWarning
from .base import IntOrParameter
from . import sequences


# families of optimizers
# pylint: disable=unused-wildcard-import,wildcard-import,too-many-lines,too-many-arguments
from .differentialevolution import *  # noqa: F403
from .es import *  # noqa: F403
from .oneshot import *  # noqa: F403
from .rescaledoneshot import *  # noqa: F403
from .recastlib import *  # noqa: F403


# # # # # optimizers # # # # #


class _OnePlusOne(base.Optimizer):
    """Simple but sometimes powerful optimization algorithm.

    We use the one-fifth adaptation rule, going back to Schumer and Steiglitz (1968).
    It was independently rediscovered by Devroye (1972) and Rechenberg (1973).
    We use asynchronous updates, so that the 1+1 can actually be parallel and even
    performs quite well in such a context - this is naturally close to 1+lambda.
    """

    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._parameters = ParametrizedOnePlusOne()
        self._sigma: float = 1

    def _internal_ask(self) -> ArrayLike:
        # pylint: disable=too-many-return-statements, too-many-branches
        noise_handling = self._parameters.noise_handling
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
        if self._parameters.crossover and self._num_ask % 2 == 1 and len(self.archive) > 2:
            return mutator.crossover(self.current_bests["pessimistic"].x, mutator.get_roulette(self.archive, num=2))
        # mutating
        mutation = self._parameters.mutation
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


class ParametrizedOnePlusOne(base.ParametrizedFamily):
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

    _optimizer_class = _OnePlusOne

    def __init__(
        self, *, noise_handling: tp.Optional[tp.Union[str, tp.Tuple[str, float]]] = None, mutation: str = "gaussian", crossover: bool = False
    ) -> None:
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
        super().__init__()


OnePlusOne = ParametrizedOnePlusOne().with_name("OnePlusOne", register=True)
NoisyOnePlusOne = ParametrizedOnePlusOne(noise_handling="random").with_name("NoisyOnePlusOne", register=True)
OptimisticNoisyOnePlusOne = ParametrizedOnePlusOne(
    noise_handling="optimistic").with_name("OptimisticNoisyOnePlusOne", register=True)
DiscreteOnePlusOne = ParametrizedOnePlusOne(mutation="discrete").with_name("DiscreteOnePlusOne", register=True)
OptimisticDiscreteOnePlusOne = ParametrizedOnePlusOne(noise_handling="optimistic", mutation="discrete").with_name(
    "OptimisticDiscreteOnePlusOne", register=True
)
NoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(noise_handling=("random", 1.0), mutation="discrete").with_name(
    "NoisyDiscreteOnePlusOne", register=True
)
DoubleFastGADiscreteOnePlusOne = ParametrizedOnePlusOne(
    mutation="doublefastga").with_name("DoubleFastGADiscreteOnePlusOne", register=True)
FastGADiscreteOnePlusOne = ParametrizedOnePlusOne(
    mutation="fastga").with_name("FastGADiscreteOnePlusOne", register=True)
DoubleFastGAOptimisticNoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(noise_handling="optimistic", mutation="doublefastga").with_name(
    "DoubleFastGAOptimisticNoisyDiscreteOnePlusOne", register=True
)
FastGAOptimisticNoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(noise_handling="optimistic", mutation="fastga").with_name(
    "FastGAOptimisticNoisyDiscreteOnePlusOne", register=True
)
FastGANoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(noise_handling="random", mutation="fastga").with_name(
    "FastGANoisyDiscreteOnePlusOne", register=True
)
PortfolioDiscreteOnePlusOne = ParametrizedOnePlusOne(
    mutation="portfolio").with_name("PortfolioDiscreteOnePlusOne", register=True)
PortfolioOptimisticNoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(noise_handling="optimistic", mutation="portfolio").with_name(
    "PortfolioOptimisticNoisyDiscreteOnePlusOne", register=True
)
PortfolioNoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(noise_handling="random", mutation="portfolio").with_name(
    "PortfolioNoisyDiscreteOnePlusOne", register=True
)
CauchyOnePlusOne = ParametrizedOnePlusOne(mutation="cauchy").with_name("CauchyOnePlusOne", register=True)
RecombiningOptimisticNoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(
    crossover=True, mutation="discrete", noise_handling="optimistic"
).with_name("RecombiningOptimisticNoisyDiscreteOnePlusOne", register=True)
RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(
    crossover=True, mutation="portfolio", noise_handling="optimistic"
).with_name("RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne", register=True)


class _CMA(base.Optimizer):

    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._parameters = ParametrizedCMA()
        self._es: Optional[cma.CMAEvolutionStrategy] = None
        # delay initialization to ease implementation of variants
        self.listx: tp.List[ArrayLike] = []
        self.listy: tp.List[float] = []
        self.to_be_asked: tp.Deque[np.ndarray] = deque()

    @property
    def es(self) -> cma.CMAEvolutionStrategy:
        if self._es is None:
            popsize = max(self.num_workers, 4 + int(3 * np.log(self.dimension)))
            diag = self._parameters.diagonal
            inopts = {"popsize": popsize, "randn": self._rng.randn, "CMA_diagonal": diag, "verbose": 0}
            self._es = cma.CMAEvolutionStrategy(x0=np.zeros(self.dimension, dtype=np.float), sigma0=self._parameters.scale, inopts=inopts)
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
                self.es.tell(self.listx, self.listy)
            except RuntimeError:
                pass
            else:
                self.listx = []
                self.listy = []

    def _internal_provide_recommendation(self) -> ArrayLike:
        if self._es is None:
            raise RuntimeError("Either ask or tell method should have been called before")
        if self.es.result.xbest is None:
            return self.current_bests["pessimistic"].x
        return self.es.result.xbest  # type: ignore


class ParametrizedCMA(base.ParametrizedFamily):
    """TODO

    Parameters
    ----------
    scale: float
        scale of the search
    diagonal: bool
        use the diagonal version of CMA (advised in big dimension)
    """

    _optimizer_class = _CMA

    def __init__(self, *, scale: float = 1.0, diagonal: bool = False) -> None:
        self.scale = scale
        self.diagonal = diagonal
        super().__init__()


CMA = ParametrizedCMA().with_name("CMA", register=True)
DiagonalCMA = ParametrizedCMA(diagonal=True).with_name("DiagonalCMA", register=True)
MilliCMA = ParametrizedCMA(scale=1e-3).with_name("MilliCMA", register=True)
MicroCMA = ParametrizedCMA(scale=1e-6).with_name("MicroCMA", register=True)


@registry.register
class EDA(base.Optimizer):
    """Test-based population-size adaptation.

    Population-size equal to lambda = 4 x dimension.
    Test by comparing the first fifth and the last fifth of the 5lambda evaluations.
    """

    # pylint: disable=too-many-instance-attributes

    @base.deprecated_init
    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.sigma = 1
        self.covariance = np.identity(self.dimension)
        self.mu = self.dimension
        self.llambda = 4 * self.dimension
        if num_workers is not None:
            self.llambda = max(self.llambda, num_workers)
        self.current_center: np.ndarray = np.zeros(self.dimension)
        # Evaluated population
        self.evaluated_population: List[ArrayLike] = []
        self.evaluated_population_sigma: List[float] = []
        self.evaluated_population_fitness: List[float] = []
        # Archive
        self.archive_fitness: List[float] = []

    def _internal_provide_recommendation(self) -> ArrayLike:  # This is NOT the naive version. We deal with noise.
        return self.current_center

    def _internal_ask_candidate(self) -> p.Parameter:
        mutated_sigma = self.sigma * np.exp(self._rng.normal(0, 1) / np.sqrt(self.dimension))
        assert len(self.current_center) == len(self.covariance), [self.dimension, self.current_center, self.covariance]
        data = mutated_sigma * self._rng.multivariate_normal(self.current_center, self.covariance)
        candidate = self.parametrization.spawn_child().set_standardized_data(data)
        candidate._meta["sigma"] = mutated_sigma
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, value: float) -> None:
        data = candidate.get_standardized_data(reference=self.parametrization)
        self.evaluated_population += [data]
        self.evaluated_population_fitness += [value]
        self.evaluated_population_sigma += [candidate._meta["sigma"]]
        if len(self.evaluated_population) >= self.llambda:
            # Sorting the population.
            sorted_pop_with_sigma_and_fitness = [
                (i, s, f)
                for f, i, s in sorted(zip(self.evaluated_population_fitness, self.evaluated_population, self.evaluated_population_sigma), key=lambda t: t[0])
            ]
            self.evaluated_population = [p[0] for p in sorted_pop_with_sigma_and_fitness]
            self.covariance = 0.1 * np.cov(np.array(self.evaluated_population).T)
            self.evaluated_population_sigma = [p[1] for p in sorted_pop_with_sigma_and_fitness]
            self.evaluated_population_fitness = [p[2] for p in sorted_pop_with_sigma_and_fitness]
            # Computing the new parent.
            arrays = [np.asarray(self.evaluated_population[i]) for i in range(self.mu)]
            self.current_center = sum(arrays) / self.mu  # type: ignore
            self.sigma = np.exp(sum([np.log(self.evaluated_population_sigma[i]) for i in range(self.mu)]) / self.mu)
            self.evaluated_population = []
            self.evaluated_population_sigma = []
            self.evaluated_population_fitness = []

    def _internal_tell_not_asked(self, candidate: p.Parameter, value: float) -> None:
        raise base.TellNotAskedNotSupportedError


@registry.register
class PCEDA(EDA):
    """Test-based population-size adaptation.

    Population-size equal to lambda = 4 x dimension.
    Test by comparing the first fifth and the last fifth of the 5lambda evaluations.
    """

    # pylint: disable=too-many-instance-attributes

    def _internal_tell_candidate(self, candidate: p.Parameter, value: float) -> None:
        self.archive_fitness += [value]
        if len(self.archive_fitness) >= 5 * self.llambda:
            first_fifth = [self.archive_fitness[i] for i in range(self.llambda)]
            last_fifth = [self.archive_fitness[i] for i in range(4 * self.llambda, 5 * self.llambda)]
            mean1 = sum(first_fifth) / float(self.llambda)
            std1 = np.std(first_fifth) / np.sqrt(self.llambda - 1)
            mean2 = sum(last_fifth) / float(self.llambda)
            std2 = np.std(last_fifth) / np.sqrt(self.llambda - 1)
            z = (mean1 - mean2) / (np.sqrt(std1 ** 2 + std2 ** 2))
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
            self.archive_fitness = []
        data = candidate.get_standardized_data(reference=self.parametrization)
        self.evaluated_population += [data]
        self.evaluated_population_fitness += [value]
        self.evaluated_population_sigma += [candidate._meta["sigma"]]
        if len(self.evaluated_population) >= self.llambda:
            # Sorting the population.
            sorted_pop_with_sigma_and_fitness = [
                (i, s, f)
                for f, i, s in sorted(zip(self.evaluated_population_fitness, self.evaluated_population, self.evaluated_population_sigma), key=lambda t: t[0])
            ]
            self.evaluated_population = [p[0] for p in sorted_pop_with_sigma_and_fitness]
            self.covariance = np.cov(np.array(self.evaluated_population).T)
            self.evaluated_population_sigma = [p[1] for p in sorted_pop_with_sigma_and_fitness]
            self.evaluated_population_fitness = [p[2] for p in sorted_pop_with_sigma_and_fitness]
            # Computing the new parent.
            arrays = [np.asarray(self.evaluated_population[i]) for i in range(self.mu)]
            self.current_center = sum(arrays) / self.mu  # type: ignore
            self.sigma = np.exp(sum([np.log(self.evaluated_population_sigma[i]) for i in range(self.mu)]) / self.mu)
            self.evaluated_population = []
            self.evaluated_population_sigma = []
            self.evaluated_population_fitness = []


@registry.register
class MPCEDA(EDA):
    """Test-based population-size adaptation.

    Population-size equal to lambda = 4 x dimension.
    Test by comparing the first fifth and the last fifth of the 5lambda evaluations.
    """

    # pylint: disable=too-many-instance-attributes

    def _internal_tell_candidate(self, candidate: p.Parameter, value: float) -> None:
        self.archive_fitness += [value]
        if len(self.archive_fitness) >= 5 * self.llambda:
            first_fifth = [self.archive_fitness[i] for i in range(self.llambda)]
            last_fifth = [self.archive_fitness[i] for i in range(4 * self.llambda, 5 * self.llambda)]
            mean1 = sum(first_fifth) / float(self.llambda)
            std1 = np.std(first_fifth) / np.sqrt(self.llambda - 1)
            mean2 = sum(last_fifth) / float(self.llambda)
            std2 = np.std(last_fifth) / np.sqrt(self.llambda - 1)
            z = (mean1 - mean2) / (np.sqrt(std1 ** 2 + std2 ** 2))
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
            self.archive_fitness = []
        data = candidate.get_standardized_data(reference=self.parametrization)
        self.evaluated_population += [data]
        self.evaluated_population_fitness += [value]
        self.evaluated_population_sigma += [candidate._meta["sigma"]]
        if len(self.evaluated_population) >= self.llambda:
            # Sorting the population.
            sorted_pop_with_sigma_and_fitness = [
                (i, s, f)
                for f, i, s in sorted(zip(self.evaluated_population_fitness, self.evaluated_population, self.evaluated_population_sigma), key=lambda t: t[0])
            ]
            self.evaluated_population = [p[0] for p in sorted_pop_with_sigma_and_fitness]
            self.covariance *= 0.9
            self.covariance += 0.1 * np.cov(np.array(self.evaluated_population).T)
            self.evaluated_population_sigma = [p[1] for p in sorted_pop_with_sigma_and_fitness]
            self.evaluated_population_fitness = [p[2] for p in sorted_pop_with_sigma_and_fitness]
            # Computing the new parent.
            arrays = [np.asarray(self.evaluated_population[i]) for i in range(self.mu)]
            self.current_center = sum(arrays) / self.mu  # type: ignore
            self.sigma = np.exp(sum([np.log(self.evaluated_population_sigma[i]) for i in range(self.mu)]) / self.mu)
            self.evaluated_population = []
            self.evaluated_population_sigma = []
            self.evaluated_population_fitness = []


@registry.register
class MEDA(EDA):
    """Test-based population-size adaptation.

    Population-size equal to lambda = 4 x dimension.
    Test by comparing the first fifth and the last fifth of the 5lambda evaluations.
    """

    # pylint: disable=too-many-instance-attributes

    def _internal_tell_candidate(self, candidate: p.Parameter, value: float) -> None:
        data = candidate.get_standardized_data(reference=self.parametrization)
        self.evaluated_population += [data]
        self.evaluated_population_fitness += [value]
        self.evaluated_population_sigma += [candidate._meta["sigma"]]
        if len(self.evaluated_population) >= self.llambda:
            # Sorting the population.
            sorted_pop_with_sigma_and_fitness = [
                (i, s, f)
                for f, i, s in sorted(zip(self.evaluated_population_fitness, self.evaluated_population, self.evaluated_population_sigma), key=lambda t: t[0])
            ]
            self.evaluated_population = [p[0] for p in sorted_pop_with_sigma_and_fitness]
            self.covariance *= 0.9
            self.covariance += 0.1 * np.cov(np.array(self.evaluated_population).T)
            self.evaluated_population_sigma = [p[1] for p in sorted_pop_with_sigma_and_fitness]
            self.evaluated_population_fitness = [p[2] for p in sorted_pop_with_sigma_and_fitness]
            # Computing the new parent.
            arrays = [np.asarray(self.evaluated_population[i]) for i in range(self.mu)]
            self.current_center = sum(arrays) / self.mu  # type: ignore
            self.sigma = np.exp(sum([np.log(self.evaluated_population_sigma[i]) for i in range(self.mu)]) / self.mu)
            self.evaluated_population = []
            self.evaluated_population_sigma = []
            self.evaluated_population_fitness = []


@registry.register
class TBPSA(base.Optimizer):
    """Test-based population-size adaptation.

    Population-size equal to lambda = 4 x dimension.
    Test by comparing the first fifth and the last fifth of the 5lambda evaluations.
    """

    # pylint: disable=too-many-instance-attributes

    @base.deprecated_init
    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.sigma = 1
        self.mu = self.dimension
        self.llambda = 4 * self.dimension
        if num_workers is not None:
            self.llambda = max(self.llambda, num_workers)
        self.current_center: np.ndarray = np.zeros(self.dimension)
        self._loss_record: List[float] = []
        # population
        self._evaluated_population: List[base.utils.Individual] = []

    def _internal_provide_recommendation(self) -> ArrayLike:  # This is NOT the naive version. We deal with noise.
        return self.current_center

    def _internal_ask_candidate(self) -> p.Parameter:
        mutated_sigma = self.sigma * np.exp(self._rng.normal(0, 1) / np.sqrt(self.dimension))
        individual = self.current_center + mutated_sigma * self._rng.normal(0, 1, self.dimension)
        candidate = self.parametrization.spawn_child().set_standardized_data(individual)
        candidate._meta["sigma"] = mutated_sigma
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, value: float) -> None:
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
        data = candidate.get_standardized_data(reference=self.parametrization)
        particle = base.utils.Individual(data)
        particle._parameters = np.array([candidate._meta["sigma"]])
        particle.value = value
        self._evaluated_population.append(particle)
        if len(self._evaluated_population) >= self.llambda:
            # Sorting the population.
            self._evaluated_population.sort(key=lambda p: p.value)
            # Computing the new parent.
            self.current_center = sum(p.x for p in self._evaluated_population[: self.mu]) / self.mu  # type: ignore
            self.sigma = np.exp(np.sum(np.log([p._parameters[0]
                                               for p in self._evaluated_population[: self.mu]])) / self.mu)
            self._evaluated_population = []

    def _internal_tell_not_asked(self, candidate: p.Parameter, value: float) -> None:
        data = candidate.get_standardized_data(reference=self.parametrization)
        sigma = np.linalg.norm(data - self.current_center) / np.sqrt(self.dimension)  # educated guess
        candidate._meta["sigma"] = sigma
        self._internal_tell_candidate(candidate, value)  # go through standard pipeline


@registry.register
class NaiveTBPSA(TBPSA):
    def _internal_provide_recommendation(self) -> ArrayLike:
        return self.current_bests["optimistic"].x


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


class PSOParticle(utils.Individual):
    """Particle for the PSO algorithm, holding relevant information (in [0,1] box)
    """

    transform = transforms.ArctanBound(0, 1).reverted()
    _eps = 0.0  # to clip to [eps, 1 - eps] for transform not defined on borders

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        x: np.ndarray,
        value: Optional[float],
        speed: np.ndarray,
        best_x: np.ndarray,
        best_value: float,
        random_state: np.random.RandomState,
    ) -> None:
        super().__init__(x)
        self.speed = speed
        self.value = value
        self.best_x = best_x
        self.best_value = best_value
        self.random_state = random_state

    @classmethod
    def from_data(cls, data: np.ndarray, random_state: np.random.RandomState) -> "PSOParticle":
        position = cls.transform.backward(data)
        speed = random_state.uniform(-1.0, 1.0, data.size)
        return cls(position, None, speed, position, float("inf"), random_state=random_state)

    def __repr__(self) -> str:
        # return f"{self.__class__.__name__}<position: {self.get_transformed_position()}, fitness: {self.value}, best: {self.best_value}>"
        return f"{self.__class__.__name__}<position: {self.x}, {self.get_transformed_position()}>"

    def mutate(self, best_x: np.ndarray, omega: float, phip: float, phig: float) -> None:
        dim = len(best_x)
        rp = self.random_state.uniform(0.0, 1.0, size=dim)
        rg = self.random_state.uniform(0.0, 1.0, size=dim)
        self.speed = omega * self.speed + phip * rp * (self.best_x - self.x) + phig * rg * (best_x - self.x)
        self.x = np.clip(self.speed + self.x, self._eps, 1 - self._eps)

    def get_transformed_position(self) -> np.ndarray:
        return self.transform.forward(self.x)


@registry.register
class PSO(base.Optimizer):
    """Partially following SPSO2011. However, no randomization of the population order.
    See: M. Zambrano-Bigiarini, M. Clerc and R. Rojas,
         Standard Particle Swarm Optimisation 2011 at CEC-2013: A baseline for future PSO improvements,
         2013 IEEE Congress on Evolutionary Computation, Cancun, 2013, pp. 2337-2344.
         https://ieeexplore.ieee.org/document/6557848
    """

    # pylint: disable=too-many-instance-attributes

    _PARTICULE = PSOParticle

    @base.deprecated_init
    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._wide = False
        if budget is not None and budget < 60:
            warnings.warn("PSO is inefficient with budget < 60", base.InefficientSettingsWarning)
        self.llambda = max(40, num_workers)
        self.population: utils.Population[PSOParticle] = utils.Population([])
        self.best_x = np.zeros(self.dimension, dtype=float)  # TODO: use current best instead?
        self.best_value = float("inf")
        self.omega = 0.5 / np.log(2.0)
        self.phip = 0.5 + np.log(2.0)
        self.phig = 0.5 + np.log(2.0)

    def _internal_ask_candidate(self) -> p.Parameter:
        # population is increased only if queue is empty (otherwise tell_not_asked does not work well at the beginning)
        if len(self.population) < self.llambda:
            param = self.parametrization
            if self._wide:
                # old initialization below seeds in the while R space, while other algorithms use normal distrib
                data = self._PARTICULE.transform.forward(self._rng.uniform(0, 1, self.dimension))
            else:
                data = param.sample().get_standardized_data(reference=param)
            self.population.extend([self._PARTICULE.from_data(data, random_state=self._rng)])
        particle = self.population.get_queued(remove=False)
        if particle.value is not None:  # particle was already initialized
            particle.mutate(best_x=self.best_x, omega=self.omega, phip=self.phip, phig=self.phig)
        candidate = self.parametrization.spawn_child().set_standardized_data(particle.get_transformed_position())
        candidate._meta["particle"] = particle
        self.population.get_queued(remove=True)
        # only remove at the last minute (safer for checkpointing)
        return candidate

    def _internal_provide_recommendation(self) -> ArrayLike:
        return self._PARTICULE.transform.forward(self.best_x)

    def _internal_tell_candidate(self, candidate: p.Parameter, value: float) -> None:
        particle: PSOParticle = candidate._meta["particle"]
        if not particle._active:
            self._internal_tell_not_asked(candidate, value)
            return
        particle.value = value
        if value < self.best_value:
            self.best_x = np.array(particle.x, copy=True)
            self.best_value = value
        if value < particle.best_value:
            particle.best_x = np.array(particle.x, copy=False)
            particle.best_value = value
        self.population.set_queued(particle)  # update when everything is well done (safer for checkpointing)

    def _internal_tell_not_asked(self, candidate: p.Parameter, value: float) -> None:
        x = candidate.get_standardized_data(reference=self.parametrization)
        if len(self.population) < self.llambda:
            particle = self._PARTICULE.from_data(x, random_state=self._rng)
            self.population.extend([particle])
        else:
            worst_part = max(iter(self.population), key=lambda p: p.best_value)  # or fitness?
            if worst_part.best_value < value:
                return  # no need to update
            particle = self._PARTICULE.from_data(x, random_state=self._rng)
            worst_part._active = False
            self.population.replace(worst_part, particle)
        # go through standard tell
        candidate._meta["particle"] = particle
        self._internal_tell_candidate(candidate, value)


@registry.register
class WidePSO(PSO):
    """Partially following SPSO2011. However, no randomization of the population order.
    This version uses a non-standard initialization with high standard deviation.
    """

    @base.deprecated_init
    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._wide = True


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

    @base.deprecated_init
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

    @base.deprecated_init
    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1, num_optims: Optional[int] = None, num_vars: Optional[List[Any]] = None, multivariate_optimizer: base.OptimizerFamily = CMA, monovariate_optimizer: base.OptimizerFamily = RandomSearch) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        if num_vars:
            if num_optims:
                assert num_optims == len(num_vars), f"The number {num_optims} of optimizers should match len(num_vars)={len(num_vars)}."
            else:
                num_optims = len(num_vars)
            assert sum(num_vars) == self.dimension, f"sum(num_vars)={sum(num_vars)} should be equal to the dimension {self.dimension}."
        else:
            if not num_optims:  # if no num_vars and no num_optims, just assume 2.
                num_optims = 2
            # num_vars not given: we will distribute variables equally.
        if num_optims > self.dimension:
            num_optims = self.dimension
        self.num_optims = num_optims
        self.optims: List[Any] = []
        self.num_vars: List[Any] = num_vars if num_vars else []
        self.parametrizations: List[Any] = []
        for i in range(self.num_optims):
            if not self.num_vars or len(self.num_vars) < i + 1:
                self.num_vars += [(self.dimension // self.num_optims) + (self.dimension % self.num_optims > i)]

            assert self.num_vars[i] >= 1, "At least one variable per optimizer."
            self.parametrizations += [p.Array(shape=(self.num_vars[i],))]
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


# Olivier: I think Jeremy will kill for doing this that way, protect me when he is back:
@registry.register
class SplitOptimizer3(SplitOptimizer):
    """Same as SplitOptimizer, but with default at 3 optimizers.
    """

    @base.deprecated_init
    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1, num_optims: int = 3, num_vars: Optional[List[Any]] = None) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers, num_optims=num_optims, num_vars=num_vars)


@registry.register
class SplitOptimizer5(SplitOptimizer):
    """Same as SplitOptimizer, but with default at 5 optimizers.
    """

    @base.deprecated_init
    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1, num_optims: int = 5, num_vars: Optional[List[Any]] = None) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers, num_optims=num_optims, num_vars=num_vars)


@registry.register
class SplitOptimizer9(SplitOptimizer):
    """Same as SplitOptimizer, but with default at 9 optimizers.
    """

    @base.deprecated_init
    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1, num_optims: int = 9, num_vars: Optional[List[Any]] = None) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers, num_optims=num_optims, num_vars=num_vars)


@registry.register
class SplitOptimizer13(SplitOptimizer):
    """Same as SplitOptimizer, but with default at 13 optimizers.
    """

    @base.deprecated_init
    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1, num_optims: int = 13, num_vars: Optional[List[Any]] = None) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers, num_optims=num_optims, num_vars=num_vars)


@registry.register
class Portfolio(base.Optimizer):
    """Passive portfolio of CMA, 2-pt DE and Scr-Hammersley."""

    @base.deprecated_init
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

    @base.deprecated_init
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

    @base.deprecated_init
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

    @base.deprecated_init
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

    @base.deprecated_init
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

    @base.deprecated_init
    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.optims = [
            CMA(self.parametrization, budget=None, num_workers=num_workers),
            TwoPointsDE(self.parametrization, budget=None, num_workers=num_workers),
        ]  # noqa: F405


@registry.register
class CMandAS2(ASCMADEthird):
    """Competence map, with algorithm selection in one of the cases (3 CMAs)."""

    @base.deprecated_init
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

    @base.deprecated_init
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

    @base.deprecated_init
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

    @base.deprecated_init
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

    @base.deprecated_init
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

    @base.deprecated_init
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

    @base.deprecated_init
    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.optims = [
            CMA(self.parametrization, budget=None, num_workers=num_workers),  # share parametrization and its rng
            MilliCMA(self.parametrization, budget=None, num_workers=num_workers),
            MicroCMA(self.parametrization, budget=None, num_workers=num_workers),
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

    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._parameters = ParametrizedBO()
        self._transform = transforms.ArctanBound(0, 1)
        self._bo: Optional[BayesianOptimization] = None
        self._fake_function = _FakeFunction()

    @property
    def bo(self) -> BayesianOptimization:
        if self._bo is None:
            params = self._parameters
            bounds = {f"x{i}": (0.0, 1.0) for i in range(self.dimension)}
            self._bo = BayesianOptimization(self._fake_function, bounds, random_state=self._rng)
            if self._parameters.gp_parameters is not None:
                self._bo.set_gp_params(**self._parameters.gp_parameters)
            # init
            init = params.initialization
            if params.middle_point:
                self._bo.probe([0.5] * self.dimension, lazy=True)
            elif init is None:
                self._bo._queue.add(self._bo._space.random_sample())
            if init is not None:
                init_budget = int(np.sqrt(self.budget) if params.init_budget is None else params.init_budget)
                init_budget -= params.middle_point
                if init_budget > 0:
                    sampler = {"Hammersley": sequences.HammersleySampler, "LHS": sequences.LHSSampler, "random": sequences.RandomSampler}[
                        init
                    ](self.dimension, budget=init_budget, scrambling=(init == "Hammersley"), random_state=self._rng)
                    for point in sampler:
                        self._bo.probe(point, lazy=True)
        return self._bo

    def _internal_ask_candidate(self) -> p.Parameter:
        params = self._parameters
        util = UtilityFunction(kind=params.utility_kind, kappa=params.utility_kappa, xi=params.utility_xi)
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
        return self._transform.backward(np.array([self.bo.max["params"][f"x{i}"] for i in range(self.dimension)]))


class ParametrizedBO(base.ParametrizedFamily):
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
    _optimizer_class = _BO

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
        assert initialization is None or initialization in ["random", "Hammersley", "LHS"], f"Unknown init {initialization}"
        self.initialization = initialization
        self.init_budget = init_budget
        self.middle_point = middle_point
        self.utility_kind = utility_kind
        self.utility_kappa = utility_kappa
        self.utility_xi = utility_xi
        self.gp_parameters = gp_parameters
        super().__init__()

    @base.deprecated_init
    def __call__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> base.Optimizer:
        gp_params = {} if self.gp_parameters is None else self.gp_parameters
        if isinstance(parametrization, p.Parameter) and gp_params.get("alpha", 0) == 0:
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
        return super().__call__(parametrization, budget, num_workers)


BO = ParametrizedBO().with_name("BO", register=True)
RBO = ParametrizedBO(initialization="random").with_name("RBO", register=True)
QRBO = ParametrizedBO(initialization="Hammersley").with_name("QRBO", register=True)
MidQRBO = ParametrizedBO(initialization="Hammersley", middle_point=True).with_name("MidQRBO", register=True)
LBO = ParametrizedBO(initialization="LHS").with_name("LBO", register=True)


@registry.register
class PBIL(base.Optimizer):
    """
    Implementation of the discrete algorithm PBIL

    https://www.ri.cmu.edu/pub_files/pub1/baluja_shumeet_1994_2/baluja_shumeet_1994_2.pdf
    """

    # pylint: disable=too-many-instance-attributes

    @base.deprecated_init
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

    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._parameters = Chaining([LHSSearch, DE], [10])  # needs a default
        # delayed initialization
        self._optimizers_: List[base.Optimizer] = []

    @property
    def _optimizers(self) -> List[base.Optimizer]:
        if not self._optimizers_:
            self._optimizers_ = []
            converter = {"num_workers": self.num_workers, "dimension": self.dimension,
                         "half": self.budget // 2 if self.budget else self.num_workers,
                         "sqrt": int(np.sqrt(self.budget)) if self.budget else self.num_workers}
            budgets = [converter[b] if isinstance(b, str) else b for b in self._parameters.budgets]
            last_budget = None if self.budget is None else self.budget - sum(budgets)
            for opt, budget in zip(self._parameters.optimizers, budgets + [last_budget]):  # type: ignore
                self._optimizers_.append(opt(self.parametrization, budget=budget, num_workers=self.num_workers))
        return self._optimizers_

    def _internal_ask_candidate(self) -> p.Parameter:
        # Which algorithm are we playing with ?
        sum_budget = 0.0
        opt = self._optimizers[0]
        for opt in self._optimizers:
            sum_budget += float("inf") if opt.budget is None else opt.budget
            if self.num_ask < sum_budget:
                break
        # if we are over budget, then use the last one...
        return opt.ask()

    def _internal_tell_candidate(self, candidate: p.Parameter, value: float) -> None:
        # Let us inform all concerned algorithms
        sum_budget = 0.0
        for opt in self._optimizers:
            sum_budget += float("inf") if opt.budget is None else opt.budget
            if self.num_tell < sum_budget:
                opt.tell(candidate, value)


class Chaining(base.ParametrizedFamily):
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
    _optimizer_class = _Chain

    def __init__(self, optimizers: tp.Sequence[tp.Union[base.OptimizerFamily, tp.Type[base.Optimizer]]],
                 budgets: tp.Sequence[tp.Union[str, int]]) -> None:
        # Either we have the budget for each algorithm, or the last algorithm uses the rest of the budget, so:
        self.budgets = tuple(budgets)
        self.optimizers = tuple(optimizers)
        assert len(self.optimizers) == len(self.budgets) + 1
        assert all(x in ("half", "dimension", "num_workers", "sqrt") or x > 0 for x in self.budgets)  # type: ignore
        super().__init__()


chainCMASQP = Chaining([CMA, SQP], ["half"]).with_name("chainCMASQP", register=True)
chainCMASQP.no_parallelization = True
chainCMAPowell = Chaining([CMA, Powell], ["half"]).with_name("chainCMAPowell", register=True)
chainCMAPowell.no_parallelization = True

chainDEwithR = Chaining([RandomSearch, DE], ["num_workers"]).with_name("chainDEwithR", register=True)
chainDEwithRsqrt = Chaining([RandomSearch, DE], ["sqrt"]).with_name("chainDEwithRsqrt", register=True)
chainDEwithRdim = Chaining([RandomSearch, DE], ["dimension"]).with_name("chainDEwithRdim", register=True)
chainDEwithR30 = Chaining([RandomSearch, DE], [30]).with_name("chainDEwithR30", register=True)
chainDEwithLHS = Chaining([LHSSearch, DE], ["num_workers"]).with_name("chainDEwithLHS", register=True)
chainDEwithLHSsqrt = Chaining([LHSSearch, DE], ["sqrt"]).with_name("chainDEwithLHSsqrt", register=True)
chainDEwithLHSdim = Chaining([LHSSearch, DE], ["dimension"]).with_name("chainDEwithLHSdim", register=True)
chainDEwithLHS30 = Chaining([LHSSearch, DE], [30]).with_name("chainDEwithLHS30", register=True)
chainDEwithMetaRecentering = Chaining([MetaRecentering, DE], ["num_workers"]).with_name("chainDEwithMetaRecentering", register=True)
chainDEwithMetaRecenteringsqrt = Chaining([MetaRecentering, DE], ["sqrt"]).with_name("chainDEwithMetaRecenteringsqrt", register=True)
chainDEwithMetaRecenteringdim = Chaining([MetaRecentering, DE], ["dimension"]).with_name("chainDEwithMetaRecenteringdim", register=True)
chainDEwithMetaRecentering30 = Chaining([MetaRecentering, DE], [30]).with_name("chainDEwithMetaRecentering30", register=True)

chainBOwithR = Chaining([RandomSearch, BO], ["num_workers"]).with_name("chainBOwithR", register=True)
chainBOwithRsqrt = Chaining([RandomSearch, BO], ["sqrt"]).with_name("chainBOwithRsqrt", register=True)
chainBOwithRdim = Chaining([RandomSearch, BO], ["dimension"]).with_name("chainBOwithRdim", register=True)
chainBOwithR30 = Chaining([RandomSearch, BO], [30]).with_name("chainBOwithR30", register=True)
chainBOwithLHS30 = Chaining([LHSSearch, BO], [30]).with_name("chainBOwithLHS30", register=True)
chainBOwithLHSsqrt = Chaining([LHSSearch, BO], ["sqrt"]).with_name("chainBOwithLHSsqrt", register=True)
chainBOwithLHSdim = Chaining([LHSSearch, BO], ["dimension"]).with_name("chainBOwithLHSdim", register=True)
chainBOwithLHS = Chaining([LHSSearch, BO], ["num_workers"]).with_name("chainBOwithLHS", register=True)
chainBOwithMetaRecentering30 = Chaining([MetaRecentering, BO], [30]).with_name("chainBOwithMetaRecentering30", register=True)
chainBOwithMetaRecenteringsqrt = Chaining([MetaRecentering, BO], ["sqrt"]).with_name("chainBOwithMetaRecenteringsqrt", register=True)
chainBOwithMetaRecenteringdim = Chaining([MetaRecentering, BO], ["dimension"]).with_name("chainBOwithMetaRecenteringdim", register=True)
chainBOwithMetaRecentering = Chaining([MetaRecentering, BO], ["num_workers"]).with_name("chainBOwithMetaRecentering", register=True)

chainPSOwithR = Chaining([RandomSearch, PSO], ["num_workers"]).with_name("chainPSOwithR", register=True)
chainPSOwithRsqrt = Chaining([RandomSearch, PSO], ["sqrt"]).with_name("chainPSOwithRsqrt", register=True)
chainPSOwithRdim = Chaining([RandomSearch, PSO], ["dimension"]).with_name("chainPSOwithRdim", register=True)
chainPSOwithR30 = Chaining([RandomSearch, PSO], [30]).with_name("chainPSOwithR30", register=True)
chainPSOwithLHS30 = Chaining([LHSSearch, PSO], [30]).with_name("chainPSOwithLHS30", register=True)
chainPSOwithLHSsqrt = Chaining([LHSSearch, PSO], ["sqrt"]).with_name("chainPSOwithLHSsqrt", register=True)
chainPSOwithLHSdim = Chaining([LHSSearch, PSO], ["dimension"]).with_name("chainPSOwithLHSdim", register=True)
chainPSOwithLHS = Chaining([LHSSearch, PSO], ["num_workers"]).with_name("chainPSOwithLHS", register=True)
chainPSOwithMetaRecentering30 = Chaining([MetaRecentering, PSO], [30]).with_name("chainPSOwithMetaRecentering30", register=True)
chainPSOwithMetaRecenteringsqrt = Chaining([MetaRecentering, PSO], ["sqrt"]).with_name("chainPSOwithMetaRecenteringsqrt", register=True)
chainPSOwithMetaRecenteringdim = Chaining([MetaRecentering, PSO], ["dimension"]).with_name("chainPSOwithMetaRecenteringdim", register=True)
chainPSOwithMetaRecentering = Chaining([MetaRecentering, PSO], ["num_workers"]).with_name("chainPSOwithMetaRecentering", register=True)

chainCMAwithR = Chaining([RandomSearch, CMA], ["num_workers"]).with_name("chainCMAwithR", register=True)
chainCMAwithRsqrt = Chaining([RandomSearch, CMA], ["sqrt"]).with_name("chainCMAwithRsqrt", register=True)
chainCMAwithRdim = Chaining([RandomSearch, CMA], ["dimension"]).with_name("chainCMAwithRdim", register=True)
chainCMAwithR30 = Chaining([RandomSearch, CMA], [30]).with_name("chainCMAwithR30", register=True)
chainCMAwithLHS30 = Chaining([LHSSearch, CMA], [30]).with_name("chainCMAwithLHS30", register=True)
chainCMAwithLHSsqrt = Chaining([LHSSearch, CMA], ["sqrt"]).with_name("chainCMAwithLHSsqrt", register=True)
chainCMAwithLHSdim = Chaining([LHSSearch, CMA], ["dimension"]).with_name("chainCMAwithLHSdim", register=True)
chainCMAwithLHS = Chaining([LHSSearch, CMA], ["num_workers"]).with_name("chainCMAwithLHS", register=True)
chainCMAwithMetaRecentering30 = Chaining([MetaRecentering, CMA], [30]).with_name("chainCMAwithMetaRecentering30", register=True)
chainCMAwithMetaRecenteringsqrt = Chaining([MetaRecentering, CMA], ["sqrt"]).with_name("chainCMAwithMetaRecenteringsqrt", register=True)
chainCMAwithMetaRecenteringdim = Chaining([MetaRecentering, CMA], ["dimension"]).with_name("chainCMAwithMetaRecenteringdim", register=True)
chainCMAwithMetaRecentering = Chaining([MetaRecentering, CMA], ["num_workers"]).with_name("chainCMAwithMetaRecentering", register=True)


@registry.register
class cGA(base.Optimizer):
    """
    Implementation of the discrete cGA algorithm

    https://pdfs.semanticscholar.org/4b0b/5733894ffc0b2968ddaab15d61751b87847a.pdf
    """

    # pylint: disable=too-many-instance-attributes

    @base.deprecated_init
    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1, arity: Optional[int] = None) -> None:
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
    @base.deprecated_init
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


class EMNA_TBPSA(TBPSA):
    """Test-based population-size adaptation with EMNA.
    """

    # pylint: disable=too-many-instance-attributes

    @base.deprecated_init
    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.sigma = 1
        self.mu = self.dimension
        self.llambda = 4 * self.dimension
        if num_workers is not None:
            self.llambda = max(self.llambda, num_workers)
        self.current_center: np.ndarray = np.zeros(self.dimension)
        self._loss_record: List[float] = []
        # population
        self._evaluated_population: List[base.utils.Individual] = []

    def _internal_provide_recommendation(self) -> ArrayLike:
        return self.current_bests["optimistic"].x  # Naive version for now

    def _internal_tell_candidate(self, candidate: p.Parameter, value: float) -> None:
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
        data = candidate.get_standardized_data(reference=self.parametrization)
        particle = base.utils.Individual(data)
        particle._parameters = np.array([candidate._meta["sigma"]])
        particle.value = value
        self._evaluated_population.append(particle)
        if len(self._evaluated_population) >= self.llambda:
            # Sorting the population.
            self._evaluated_population.sort(key=lambda p: p.value)
            # Computing the new parent.
            self.current_center = sum(p.x for p in self._evaluated_population[: self.mu]) / self.mu  # type: ignore
            # EMNA update
            t1 = [(self._evaluated_population[i].x - self.current_center)**2 for i in range(self.mu)]
            self.sigma = np.sqrt(sum(t1) / (self.mu))
            imp = max(1, (np.log(self.llambda) / 2)**(1 / self.dimension))
            if self.num_workers / self.dimension > 16:
                self.sigma /= imp
            self._evaluated_population = []


@registry.register
class Shiva(NGO):
    """Nevergrad optimizer by competence map. You might modify this one for designing youe own competence map."""

    @base.deprecated_init
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
