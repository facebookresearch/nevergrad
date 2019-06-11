# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, List, Dict, Tuple, Deque, Union, Callable, Any
from collections import defaultdict, deque
import warnings
import cma
import numpy as np
from bayes_opt import UtilityFunction
from bayes_opt import BayesianOptimization
from ..instrumentation import transforms
from ..instrumentation import Instrumentation
from . import utils
from . import base
from . import mutations
from .base import registry
from .base import InefficientSettingsWarning
from . import sequences
# families of optimizers
# pylint: disable=unused-wildcard-import,wildcard-import, too-many-lines
from .differentialevolution import *  # noqa: F403
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

    def __init__(self, instrumentation: Union[int, Instrumentation], budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        self._parameters = ParametrizedOnePlusOne()
        self._sigma: float = 1

    def _internal_ask(self) -> base.ArrayLike:
        # pylint: disable=too-many-return-statements, too-many-branches
        noise_handling = self._parameters.noise_handling
        if not self._num_ask:
            return np.zeros(self.dimension)  # type: ignore
        # for noisy version
        if noise_handling is not None:
            limit = (.05 if isinstance(noise_handling, str) else noise_handling[1]) * len(self.archive) ** 3
            strategy = noise_handling if isinstance(noise_handling, str) else noise_handling[0]
            if self._num_ask <= limit:
                if strategy in ["cubic", "random"]:
                    idx = self.random_state.choice(len(self.archive))
                    return np.frombuffer(list(self.archive.bytesdict.keys())[idx])  # type: ignore
                elif strategy == "optimistic":
                    return self.current_bests["optimistic"].x
        # crossover
        mutator = mutations.Mutator(self.random_state)
        if self._parameters.crossover and self._num_ask % 2 == 1 and len(self.archive) > 2:
            return mutator.crossover(self.current_bests["pessimistic"].x,
                                     mutator.get_roulette(self.archive, num=2))
        # mutating
        mutation = self._parameters.mutation
        if mutation == "gaussian":  # standard case
            return self.current_bests["pessimistic"].x + self._sigma * self.random_state.normal(0, 1, self.dimension)  # type: ignore
        elif mutation == "cauchy":
            return self.current_bests["pessimistic"].x + self._sigma * self.random_state.standard_cauchy(self.dimension)  # type: ignore
        elif mutation == "crossover":
            if self._num_ask % 2 == 0 or len(self.archive) < 3:
                return mutator.portfolio_discrete_mutation(self.current_bests["pessimistic"].x)
            else:
                return mutator.crossover(self.current_bests["pessimistic"].x,
                                         mutator.get_roulette(self.archive, num=2))
        else:
            func: Callable[[base.ArrayLike], base.ArrayLike] = {  # type: ignore
                "discrete": mutator.discrete_mutation,
                "fastga": mutator.doerr_discrete_mutation,
                "doublefastga": mutator.doubledoerr_discrete_mutation,
                "portfolio": mutator.portfolio_discrete_mutation}[mutation]
            return func(self.current_bests["pessimistic"].x)

    def _internal_tell(self, x: base.ArrayLike, value: float) -> None:
        # only used for cauchy and gaussian
        self._sigma *= 2. if value <= self.current_bests["pessimistic"].mean else .84


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

    def __init__(self, *, noise_handling: Optional[Union[str, Tuple[str, float]]] = None,
                 mutation: str = "gaussian", crossover: bool = False) -> None:
        if noise_handling is not None:
            if isinstance(noise_handling, str):
                assert noise_handling in ["random", "optimistic"], f"Unkwnown noise handling: '{noise_handling}'"
            else:
                assert isinstance(noise_handling, tuple), "noise_handling must be a string or  a tuple of type (strategy, factor)"
                assert noise_handling[1] > 0., "the factor must be a float greater than 0"
                assert noise_handling[0] in ["random", "optimistic"], f"Unkwnown noise handling: '{noise_handling}'"
        assert mutation in ["gaussian", "cauchy", "discrete", "fastga", "doublefastga", "portfolio"], f"Unkwnown mutation: '{mutation}'"
        self.noise_handling = noise_handling
        self.mutation = mutation
        self.crossover = crossover
        super().__init__()


OnePlusOne = ParametrizedOnePlusOne().with_name("OnePlusOne", register=True)
NoisyOnePlusOne = ParametrizedOnePlusOne(noise_handling="random").with_name("NoisyOnePlusOne", register=True)
OptimisticNoisyOnePlusOne = ParametrizedOnePlusOne(noise_handling="optimistic").with_name("OptimisticNoisyOnePlusOne", register=True)
DiscreteOnePlusOne = ParametrizedOnePlusOne(mutation="discrete").with_name("DiscreteOnePlusOne", register=True)
OptimisticDiscreteOnePlusOne = ParametrizedOnePlusOne(
    noise_handling="optimistic", mutation="discrete").with_name("OptimisticDiscreteOnePlusOne", register=True)
NoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(
    noise_handling=("random", 1.), mutation="discrete").with_name("NoisyDiscreteOnePlusOne", register=True)
DoubleFastGADiscreteOnePlusOne = ParametrizedOnePlusOne(mutation="doublefastga").with_name("DoubleFastGADiscreteOnePlusOne", register=True)
FastGADiscreteOnePlusOne = ParametrizedOnePlusOne(
    mutation="fastga").with_name("FastGADiscreteOnePlusOne", register=True)
DoubleFastGAOptimisticNoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(
    noise_handling="optimistic", mutation="doublefastga").with_name("DoubleFastGAOptimisticNoisyDiscreteOnePlusOne", register=True)
FastGAOptimisticNoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(
    noise_handling="optimistic", mutation="fastga").with_name("FastGAOptimisticNoisyDiscreteOnePlusOne", register=True)
FastGANoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(
    noise_handling="random", mutation="fastga").with_name("FastGANoisyDiscreteOnePlusOne", register=True)
PortfolioDiscreteOnePlusOne = ParametrizedOnePlusOne(mutation="portfolio").with_name("PortfolioDiscreteOnePlusOne", register=True)
PortfolioOptimisticNoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(
    noise_handling="optimistic", mutation="portfolio").with_name("PortfolioOptimisticNoisyDiscreteOnePlusOne", register=True)
PortfolioNoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(
    noise_handling="random", mutation="portfolio").with_name("PortfolioNoisyDiscreteOnePlusOne", register=True)
CauchyOnePlusOne = ParametrizedOnePlusOne(mutation="cauchy").with_name("CauchyOnePlusOne", register=True)
RecombiningOptimisticNoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(
    crossover=True, mutation="discrete", noise_handling="optimistic").with_name(
        "RecombiningOptimisticNoisyDiscreteOnePlusOne", register=True)
RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne = ParametrizedOnePlusOne(
    crossover=True, mutation="portfolio", noise_handling="optimistic").with_name(
        "RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne", register=True)


class _CMA(base.Optimizer):
    def __init__(self, instrumentation: Union[int, Instrumentation], budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        self._parameters = ParametrizedCMA()
        self._es: Optional[cma.CMAEvolutionStrategy] = None
        # delay initialization to ease implementation of variants
        self.listx: List[base.ArrayLike] = []
        self.listy: List[float] = []
        self.to_be_asked: Deque[np.ndarray] = deque()

    @property
    def es(self) -> cma.CMAEvolutionStrategy:
        if self._es is None:
            popsize = max(self.num_workers, 4 + int(3 * np.log(self.dimension)))
            diag = self._parameters.diagonal
            inopts = {"popsize": popsize, "randn": self.random_state.randn, "CMA_diagonal": diag, "verbose": 0}
            self._es = cma.CMAEvolutionStrategy(x0=np.zeros(self.dimension, dtype=np.float),
                                                sigma0=self._parameters.scale, inopts=inopts)
        return self._es

    def _internal_ask(self) -> base.ArrayLike:
        if not self.to_be_asked:
            self.to_be_asked.extend(self.es.ask())
        return self.to_be_asked.popleft()

    def _internal_tell(self, x: base.ArrayLike, value: float) -> None:
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

    def _internal_provide_recommendation(self) -> base.ArrayLike:
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

    def __init__(self, *, scale: float = 1., diagonal: bool = False) -> None:
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

    def __init__(self, instrumentation: Union[int, Instrumentation], budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        self.sigma = 1
        self.covariance = np.identity(self.dimension)
        self.mu = self.dimension
        self.llambda = 4 * self.dimension
        if num_workers is not None:
            self.llambda = max(self.llambda, num_workers)
        self.current_center: np.ndarray = np.zeros(self.dimension)
        # Evaluated population
        self.evaluated_population: List[base.ArrayLike] = []
        self.evaluated_population_sigma: List[float] = []
        self.evaluated_population_fitness: List[float] = []
        # Unevaluated population
        self.unevaluated_population: List[base.ArrayLike] = []
        self.unevaluated_population_sigma: List[float] = []
        # Archive
        self.archive_fitness: List[float] = []

    def _internal_provide_recommendation(self) -> base.ArrayLike:  # This is NOT the naive version. We deal with noise.
        return self.current_center

    def _internal_ask(self) -> base.ArrayLike:
        mutated_sigma = self.sigma * np.exp(self.random_state.normal(0, 1) / np.sqrt(self.dimension))
        assert len(self.current_center) == len(self.covariance), [self.dimension, self.current_center, self.covariance]
        individual = tuple(mutated_sigma * self.random_state.multivariate_normal(self.current_center, self.covariance))
        self.unevaluated_population_sigma += [mutated_sigma]
        self.unevaluated_population += [tuple(individual)]
        return individual

    def _internal_tell(self, x: base.ArrayLike, value: float) -> None:
        idx = self.unevaluated_population.index(tuple(x))
        self.evaluated_population += [x]
        self.evaluated_population_fitness += [value]
        self.evaluated_population_sigma += [self.unevaluated_population_sigma[idx]]
        del self.unevaluated_population[idx]
        del self.unevaluated_population_sigma[idx]
        if len(self.evaluated_population) >= self.llambda:
            # Sorting the population.
            sorted_pop_with_sigma_and_fitness = [(i, s, f) for f, i, s in sorted(
                zip(self.evaluated_population_fitness, self.evaluated_population, self.evaluated_population_sigma))]
            self.evaluated_population = [p[0] for p in sorted_pop_with_sigma_and_fitness]
            self.covariance = .1 * np.cov(np.array(self.evaluated_population).T)
            self.evaluated_population_sigma = [p[1] for p in sorted_pop_with_sigma_and_fitness]
            self.evaluated_population_fitness = [p[2] for p in sorted_pop_with_sigma_and_fitness]
            # Computing the new parent.
            self.current_center = sum([np.asarray(self.evaluated_population[i]) for i in range(self.mu)]) / self.mu  # type: ignore
            self.sigma = np.exp(sum([np.log(self.evaluated_population_sigma[i]) for i in range(self.mu)]) / self.mu)
            self.evaluated_population = []
            self.evaluated_population_sigma = []
            self.evaluated_population_fitness = []

    def _internal_tell_not_asked(self, candidate: base.Candidate, value: float) -> None:
        raise base.TellNotAskedNotSupportedError


@registry.register
class PCEDA(EDA):
    """Test-based population-size adaptation.

    Population-size equal to lambda = 4 x dimension.
    Test by comparing the first fifth and the last fifth of the 5lambda evaluations.
    """
    # pylint: disable=too-many-instance-attributes

    def _internal_tell(self, x: base.ArrayLike, value: float) -> None:
        self.archive_fitness += [value]
        if len(self.archive_fitness) >= 5 * self.llambda:
            first_fifth = [self.archive_fitness[i] for i in range(self.llambda)]
            last_fifth = [self.archive_fitness[i] for i in range(4 * self.llambda, 5 * self.llambda)]
            mean1 = sum(first_fifth) / float(self.llambda)
            std1 = np.std(first_fifth) / np.sqrt(self.llambda - 1)
            mean2 = sum(last_fifth) / float(self.llambda)
            std2 = np.std(last_fifth) / np.sqrt(self.llambda - 1)
            z = (mean1 - mean2) / (np.sqrt(std1**2 + std2**2))
            if z < 2.:
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
        idx = self.unevaluated_population.index(tuple(x))
        self.evaluated_population += [x]
        self.evaluated_population_fitness += [value]
        self.evaluated_population_sigma += [self.unevaluated_population_sigma[idx]]
        del self.unevaluated_population[idx]
        del self.unevaluated_population_sigma[idx]
        if len(self.evaluated_population) >= self.llambda:
            # Sorting the population.
            sorted_pop_with_sigma_and_fitness = [(i, s, f) for f, i, s in sorted(
                zip(self.evaluated_population_fitness, self.evaluated_population, self.evaluated_population_sigma))]
            self.evaluated_population = [p[0] for p in sorted_pop_with_sigma_and_fitness]
            self.covariance = np.cov(np.array(self.evaluated_population).T)
            self.evaluated_population_sigma = [p[1] for p in sorted_pop_with_sigma_and_fitness]
            self.evaluated_population_fitness = [p[2] for p in sorted_pop_with_sigma_and_fitness]
            # Computing the new parent.
            self.current_center = sum([np.asarray(self.evaluated_population[i]) for i in range(self.mu)]) / self.mu  # type: ignore
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

    def _internal_tell(self, x: base.ArrayLike, value: float) -> None:
        self.archive_fitness += [value]
        if len(self.archive_fitness) >= 5 * self.llambda:
            first_fifth = [self.archive_fitness[i] for i in range(self.llambda)]
            last_fifth = [self.archive_fitness[i] for i in range(4 * self.llambda, 5 * self.llambda)]
            mean1 = sum(first_fifth) / float(self.llambda)
            std1 = np.std(first_fifth) / np.sqrt(self.llambda - 1)
            mean2 = sum(last_fifth) / float(self.llambda)
            std2 = np.std(last_fifth) / np.sqrt(self.llambda - 1)
            z = (mean1 - mean2) / (np.sqrt(std1**2 + std2**2))
            if z < 2.:
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
        idx = self.unevaluated_population.index(tuple(x))
        self.evaluated_population += [x]
        self.evaluated_population_fitness += [value]
        self.evaluated_population_sigma += [self.unevaluated_population_sigma[idx]]
        del self.unevaluated_population[idx]
        del self.unevaluated_population_sigma[idx]
        if len(self.evaluated_population) >= self.llambda:
            # Sorting the population.
            sorted_pop_with_sigma_and_fitness = [(i, s, f) for f, i, s in sorted(
                zip(self.evaluated_population_fitness, self.evaluated_population, self.evaluated_population_sigma))]
            self.evaluated_population = [p[0] for p in sorted_pop_with_sigma_and_fitness]
            self.covariance *= .9
            self.covariance += .1 * np.cov(np.array(self.evaluated_population).T)
            self.evaluated_population_sigma = [p[1] for p in sorted_pop_with_sigma_and_fitness]
            self.evaluated_population_fitness = [p[2] for p in sorted_pop_with_sigma_and_fitness]
            # Computing the new parent.
            self.current_center = sum([np.asarray(self.evaluated_population[i]) for i in range(self.mu)]) / self.mu  # type: ignore
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

    def _internal_tell(self, x: base.ArrayLike, value: float) -> None:
        idx = self.unevaluated_population.index(tuple(x))
        self.evaluated_population += [x]
        self.evaluated_population_fitness += [value]
        self.evaluated_population_sigma += [self.unevaluated_population_sigma[idx]]
        del self.unevaluated_population[idx]
        del self.unevaluated_population_sigma[idx]
        if len(self.evaluated_population) >= self.llambda:
            # Sorting the population.
            sorted_pop_with_sigma_and_fitness = [(i, s, f) for f, i, s in sorted(
                zip(self.evaluated_population_fitness, self.evaluated_population, self.evaluated_population_sigma))]
            self.evaluated_population = [p[0] for p in sorted_pop_with_sigma_and_fitness]
            self.covariance *= .9
            self.covariance += .1 * np.cov(np.array(self.evaluated_population).T)
            self.evaluated_population_sigma = [p[1] for p in sorted_pop_with_sigma_and_fitness]
            self.evaluated_population_fitness = [p[2] for p in sorted_pop_with_sigma_and_fitness]
            # Computing the new parent.
            self.current_center = sum([np.asarray(self.evaluated_population[i]) for i in range(self.mu)]) / self.mu  # type: ignore
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

    def __init__(self, instrumentation: Union[int, Instrumentation], budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        self.sigma = 1
        self.mu = self.dimension
        self.llambda = 4 * self.dimension
        if num_workers is not None:
            self.llambda = max(self.llambda, num_workers)
        self.current_center: np.ndarray = np.zeros(self.dimension)
        self._loss_record: List[float] = []
        # population
        self._evaluated_population: List[base.utils.Individual] = []
        self._unevaluated_population: Dict[bytes, base.utils.Individual] = {}

    def _internal_provide_recommendation(self) -> base.ArrayLike:  # This is NOT the naive version. We deal with noise.
        return self.current_center

    def _internal_ask(self) -> base.ArrayLike:
        mutated_sigma = self.sigma * np.exp(self.random_state.normal(0, 1) / np.sqrt(self.dimension))
        individual = self.current_center + mutated_sigma * self.random_state.normal(0, 1, self.dimension)
        part = base.utils.Individual(individual)
        part._parameters = np.array([mutated_sigma])
        self._unevaluated_population[individual.tobytes()] = part
        return individual  # type: ignore

    def _internal_tell(self, x: base.ArrayLike, value: float) -> None:
        self._loss_record += [value]
        if len(self._loss_record) >= 5 * self.llambda:
            first_fifth = self._loss_record[: self.llambda]
            last_fifth = self._loss_record[-self.llambda:]
            means = [sum(fitnesses) / float(self.llambda) for fitnesses in [first_fifth, last_fifth]]
            stds = [np.std(fitnesses) / np.sqrt(self.llambda - 1) for fitnesses in [first_fifth, last_fifth]]
            z = (means[0] - means[1]) / (np.sqrt(stds[0]**2 + stds[1]**2))
            if z < 2.:
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
        x = np.array(x, copy=False)
        x_bytes = x.tobytes()
        particle = self._unevaluated_population[x_bytes]
        particle.value = value
        self._evaluated_population.append(particle)
        if len(self._evaluated_population) >= self.llambda:
            # Sorting the population.
            self._evaluated_population.sort(key=lambda p: p.value)
            # Computing the new parent.
            self.current_center = sum(p.x for p in self._evaluated_population[:self.mu]) / self.mu  # type: ignore
            self.sigma = np.exp(np.sum(np.log([p._parameters[0] for p in self._evaluated_population[:self.mu]])) / self.mu)
            self._evaluated_population = []
        del self._unevaluated_population[x_bytes]

    def _internal_tell_not_asked(self, candidate: base.Candidate, value: float) -> None:
        x = candidate.data
        sigma = np.linalg.norm(x - self.current_center) / np.sqrt(self.dimension)  # educated guess
        part = base.utils.Individual(x)
        part._parameters = np.array([sigma])
        self._unevaluated_population[x.tobytes()] = part
        self._internal_tell_candidate(candidate, value)  # go through standard pipeline


@registry.register
class NaiveTBPSA(TBPSA):

    def _internal_provide_recommendation(self) -> base.ArrayLike:
        return self.current_bests["optimistic"].x


@registry.register
class NoisyBandit(base.Optimizer):
    """UCB.
    This is upper confidence bound (adapted to minimization),
    with very poor parametrization; in particular, the logarithmic term is set to zero.
    Infinite arms: we add one arm when `20 * #ask >= #arms ** 3`.
    """

    def _internal_ask(self) -> base.ArrayLike:
        if 20 * self._num_ask >= len(self.archive) ** 3:
            return self.random_state.normal(0, 1, self.dimension)  # type: ignore
        if self.random_state.choice([True, False]):
            # numpy does not accept choice on list of tuples, must choose index instead
            idx = self.random_state.choice(len(self.archive))
            return np.frombuffer(list(self.archive.bytesdict.keys())[idx])  # type: ignore
        return self.current_bests["optimistic"].x


class PSOParticle(utils.Individual):
    """Particle for the PSO algorithm, holding relevant information
    """

    transform = transforms.ArctanBound(0, 1).reverted()
    _eps = 0.  # to clip to [eps, 1 - eps] for transform not defined on borders

    # pylint: disable=too-many-arguments
    def __init__(self, x: np.ndarray, value: Optional[float], speed: np.ndarray,
                 best_x: np.ndarray, best_value: float, random_state: np.random.RandomState) -> None:
        super().__init__(x)
        self.speed = speed
        self.value = value
        self.best_x = best_x
        self.best_value = best_value
        self.random_state = random_state

    @classmethod
    def random_initialization(cls, dimension: int, random_state: np.random.RandomState) -> 'PSOParticle':
        position = random_state.uniform(0., 1., dimension)
        speed = random_state.uniform(-1., 1., dimension)
        return cls(position, None, speed, position, float("inf"), random_state=random_state)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<position: {self.get_transformed_position()}, fitness: {self.value}, best: {self.best_value}>"

    def mutate(self, best_x: np.ndarray, omega: float, phip: float, phig: float) -> None:
        dim = len(best_x)
        rp = self.random_state.uniform(0., 1., size=dim)
        rg = self.random_state.uniform(0., 1., size=dim)
        self.speed = (omega * self.speed
                      + phip * rp * (self.best_x - self.x)
                      + phig * rg * (best_x - self.x))
        self.x = np.clip(self.speed + self.x, self._eps, 1 - self._eps)

    def get_transformed_position(self) -> np.ndarray:
        return self.transform.forward(self.x)


@registry.register
class PSO(base.Optimizer):
    """Partially following SPSO2011. However, no randomization of the population order.
    """
    # pylint: disable=too-many-instance-attributes

    _PARTICULE = PSOParticle

    def __init__(self, instrumentation: Union[int, Instrumentation], budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        if budget is not None and budget < 60:
            warnings.warn("PSO is inefficient with budget < 60", base.InefficientSettingsWarning)
        self.llambda = max(40, num_workers)
        self.population: utils.Population[PSOParticle] = utils.Population([])
        self.best_x = np.zeros(self.dimension, dtype=float)  # TODO: use current best instead?
        self.best_value = float("inf")
        self.omega = 0.5 / np.log(2.)
        self.phip = 0.5 + np.log(2.)
        self.phig = 0.5 + np.log(2.)

    def _internal_ask_candidate(self) -> base.Candidate:
        # population is increased only if queue is empty (otherwise tell_not_asked does not work well at the beginning)
        if self.population.is_queue_empty() and len(self.population) < self.llambda:
            additional = [self._PARTICULE.random_initialization(self.dimension, random_state=self.random_state)
                          for _ in range(self.llambda - len(self.population))]
            self.population.extend(additional)
        particle = self.population.get_queued(remove=False)
        if particle.value is not None:  # particle was already initialized
            particle.mutate(best_x=self.best_x, omega=self.omega, phip=self.phip, phig=self.phig)
        candidate = self.create_candidate.from_data(particle.get_transformed_position())
        candidate._meta["particle"] = particle
        self.population.get_queued(remove=True)
        # only remove at the last minute (safer for checkpointing)
        return candidate

    def _internal_provide_recommendation(self) -> base.ArrayLike:
        return self._PARTICULE.transform.forward(self.best_x)

    def _internal_tell_candidate(self, candidate: base.Candidate, value: float) -> None:
        particle: PSOParticle = candidate._meta["particle"]
        if not particle._active:
            self._internal_tell_not_asked(candidate, value)
            return
        x = candidate.data
        point = particle.get_transformed_position()
        assert np.array_equal(x, point), f"{x} vs {point} - from population: {self.population}"
        particle.value = value
        if value < self.best_value:
            self.best_x = np.array(particle.x, copy=True)
            self.best_value = value
        if value < particle.best_value:
            particle.best_x = np.array(particle.x, copy=False)
            particle.best_value = value
        self.population.set_queued(particle)  # update when everything is well done (safer for checkpointing)

    def _internal_tell_not_asked(self, candidate: base.Candidate, value: float) -> None:
        x = candidate.data
        if len(self.population) < self.llambda:
            particle = self._PARTICULE.random_initialization(self.dimension, random_state=self.random_state)
            particle.x = self._PARTICULE.transform.backward(x)
            self.population.extend([particle])
        else:
            worst_part = max(iter(self.population), key=lambda p: p.best_value)  # or fitness?
            if worst_part.best_value < value:
                return  # no need to update
            particle = self._PARTICULE.random_initialization(self.dimension, random_state=self.random_state)
            particle.x = self._PARTICULE.transform.backward(x)
            worst_part._active = False
            self.population.replace(worst_part, particle)
        # go through standard pipeline
        c2 = self._internal_ask_candidate()
        self._internal_tell_candidate(c2, value)


@registry.register
class SPSA(base.Optimizer):
    # pylint: disable=too-many-instance-attributes
    ''' The First order SPSA algorithm as shown in [1,2,3], with implementation details
    from [4,5].

    1) https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation
    2) https://www.chessprogramming.org/SPSA
    3) Spall, James C. "Multivariate stochastic approximation using a simultaneous perturbation gradient approximation."
       IEEE transactions on automatic control 37.3 (1992): 332-341.
    4) Section 7.5.2 in "Introduction to Stochastic Search and Optimization: Estimation, Simulation and Control" by James C. Spall.
    5) Pushpendre Rastogi, Jingyi Zhu, James C. Spall CISS (2016).
       Efficient implementation of Enhanced Adaptive Simultaneous Perturbation Algorithms.
    '''
    no_parallelization = True

    def __init__(self, instrumentation: Union[int, Instrumentation], budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        self.init = True
        self.idx = 0
        self.delta = float('nan')
        self.ym: Optional[np.ndarray] = None
        self.yp: Optional[np.ndarray] = None
        self.t: np.ndarray = np.zeros(self.dimension)
        self.avg: np.ndarray = np.zeros(self.dimension)
        # Set A, a, c according to the practical implementation
        # guidelines in the ISSO book.
        self.A = (10 if budget is None else max(10, budget // 20))
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
        'c_k determines the pertubation.'
        return self.c / (k // 2 + 1)**0.101

    def _ak(self, k: int) -> float:
        'a_k is the learning rate.'
        return self.a / (k // 2 + 1 + self.A)**0.602

    def _internal_ask(self) -> base.ArrayLike:
        k = self.idx
        if k % 2 == 0:
            if not self.init:
                assert self.yp is not None and self.ym is not None
                self.t -= (self._ak(k) * (self.yp - self.ym) / 2 / self._ck(k)) * self.delta
                self.avg += (self.t - self.avg) / (k // 2 + 1)
            self.delta = 2 * self.random_state.randint(2, size=self.dimension) - 1
            return self.t - self._ck(k) * self.delta  # type:ignore
        return self.t + self._ck(k) * self.delta  # type: ignore

    def _internal_tell(self, x: base.ArrayLike, value: float) -> None:
        setattr(self, ('ym' if self.idx % 2 == 0 else 'yp'), np.array(value, copy=True))
        self.idx += 1
        if self.init and self.yp is not None and self.ym is not None:
            self.init = False

    def _internal_provide_recommendation(self) -> base.ArrayLike:
        return self.avg


@registry.register
class Portfolio(base.Optimizer):
    """Passive portfolio of CMA, 2-pt DE and Scr-Hammersley."""

    def __init__(self, instrumentation: Union[int, Instrumentation], budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        assert budget is not None
        self.optims = [CMA(instrumentation, budget // 3 + (budget % 3 > 0), num_workers),
                       TwoPointsDE(instrumentation, budget // 3 + (budget % 3 > 1), num_workers),  # noqa: F405
                       ScrHammersleySearch(instrumentation, budget // 3, num_workers)]  # noqa: F405
        if budget < 12 * num_workers:
            self.optims = [ScrHammersleySearch(instrumentation, budget, num_workers)]  # noqa: F405
        self.who_asked: Dict[Tuple[float, ...], List[int]] = defaultdict(list)
        for optim in self.optims:
            optim._random_state = self.random_state  # share the state

    def _internal_ask_candidate(self) -> base.Candidate:
        optim_index = self._num_ask % len(self.optims)
        individual = self.optims[optim_index].ask()
        self.who_asked[tuple(individual.data)] += [optim_index]
        return individual

    def _internal_tell_candidate(self, candidate: base.Candidate, value: float) -> None:
        tx = tuple(candidate.data)
        optim_index = self.who_asked[tx][0]
        del self.who_asked[tx][0]
        self.optims[optim_index].tell(candidate, value)

    def _internal_provide_recommendation(self) -> base.ArrayLike:
        return self.current_bests["pessimistic"].x

    def _internal_tell_not_asked(self, candidate: base.Candidate, value: float) -> None:
        raise base.TellNotAskedNotSupportedError


@registry.register
class ParaPortfolio(Portfolio):
    """Passive portfolio of CMA, 2-pt DE, PSO, SQP and Scr-Hammersley."""

    def __init__(self, instrumentation: Union[int, Instrumentation], budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
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
        self.optims = [CMA(instrumentation, num_workers=nw1),
                       TwoPointsDE(instrumentation, num_workers=nw2),  # noqa: F405
                       PSO(instrumentation, num_workers=nw3),
                       SQP(instrumentation, 1),  # noqa: F405
                       ScrHammersleySearch(instrumentation, budget=(budget // len(self.which_optim)) * nw4)  # noqa: F405
                       ]
        for optim in self.optims:
            optim._random_state = self.random_state  # share the state
        self.who_asked: Dict[Tuple[float, ...], List[int]] = defaultdict(list)

    def _internal_ask_candidate(self) -> base.Candidate:
        optim_index = self.which_optim[self._num_ask % len(self.which_optim)]
        individual = self.optims[optim_index].ask()
        self.who_asked[tuple(individual.data)] += [optim_index]
        return individual


@registry.register
class ParaSQPCMA(ParaPortfolio):
    """Passive portfolio of CMA and many SQP."""

    def __init__(self, instrumentation: Union[int, Instrumentation], budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        assert budget is not None
        nw = num_workers // 2
        self.which_optim = [0] * nw
        for i in range(num_workers - nw):
            self.which_optim += [i + 1]
        assert len(self.which_optim) == num_workers
        # b1, b2, b3, b4, b5 = intshare(budget, 5)
        self.optims = [CMA(instrumentation, num_workers=nw)]
        for i in range(num_workers - nw):
            self.optims += [SQP(instrumentation, 1)]  # noqa: F405
            if i > 0:
                self.optims[-1].initial_guess = self.random_state.normal(0, 1, self.dimension)  # type: ignore
        self.who_asked: Dict[Tuple[float, ...], List[int]] = defaultdict(list)
        for optim in self.optims:
            optim._random_state = self.random_state  # share the state


@registry.register
class ASCMADEthird(Portfolio):
    """Algorithm selection, with CMA and Lhs-DE. Active selection at 1/3."""

    def __init__(self, instrumentation: Union[int, Instrumentation], budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        assert budget is not None
        self.optims = [CMA(instrumentation, budget=None, num_workers=num_workers),
                       LhsDE(instrumentation, budget=None, num_workers=num_workers)]  # noqa: F405
        for optim in self.optims:
            optim._random_state = self.random_state  # share the state
        self.who_asked: Dict[Tuple[float, ...], List[int]] = defaultdict(list)
        self.budget_before_choosing = budget // 3
        self.best_optim = -1

    def _internal_ask_candidate(self) -> base.Candidate:
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
        individual = self.optims[optim_index].ask()
        self.who_asked[tuple(individual.data)] += [optim_index]
        return individual


@registry.register
class ASCMADEQRthird(ASCMADEthird):
    """Algorithm selection, with CMA, ScrHalton and Lhs-DE. Active selection at 1/3."""

    def __init__(self, instrumentation: Union[int, Instrumentation], budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        self.optims = [CMA(instrumentation, budget=None, num_workers=num_workers),
                       LhsDE(instrumentation, budget=None, num_workers=num_workers),  # noqa: F405
                       ScrHaltonSearch(instrumentation, budget=None, num_workers=num_workers)]  # noqa: F405
        for optim in self.optims:
            optim._random_state = self.random_state  # share the state


@registry.register
class ASCMA2PDEthird(ASCMADEQRthird):
    """Algorithm selection, with CMA and 2pt-DE. Active selection at 1/3."""

    def __init__(self, instrumentation: Union[int, Instrumentation], budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        self.optims = [CMA(instrumentation, budget=None, num_workers=num_workers),
                       TwoPointsDE(instrumentation, budget=None, num_workers=num_workers)]  # noqa: F405
        for optim in self.optims:
            optim._random_state = self.random_state  # share the state


@registry.register
class CMandAS2(ASCMADEthird):
    """Competence map, with algorithm selection in one of the cases (3 CMAs)."""

    def __init__(self, instrumentation: Union[int, Instrumentation], budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        self.optims = [TwoPointsDE(instrumentation, budget=None, num_workers=num_workers)]  # noqa: F405
        assert budget is not None
        self.budget_before_choosing = 2 * budget
        if budget < 201:
            self.optims = [OnePlusOne(instrumentation, budget=None, num_workers=num_workers)]
        if budget > 50 * self.dimension or num_workers < 30:
            self.optims = [CMA(instrumentation, budget=None, num_workers=num_workers),
                           CMA(instrumentation, budget=None, num_workers=num_workers),
                           CMA(instrumentation, budget=None, num_workers=num_workers)]
            self.budget_before_choosing = budget // 10
        for optim in self.optims:
            optim._random_state = self.random_state  # share the state


@registry.register
class CMandAS(CMandAS2):
    """Competence map, with algorithm selection in one of the cases (2 CMAs)."""

    def __init__(self, instrumentation: Union[int, Instrumentation], budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        self.optims = [TwoPointsDE(instrumentation, budget=None, num_workers=num_workers)]  # noqa: F405
        assert budget is not None
        self.budget_before_choosing = 2 * budget
        if budget < 201:
            self.optims = [OnePlusOne(instrumentation, budget=None, num_workers=num_workers)]
            self.budget_before_choosing = 2 * budget
        if budget > 50 * self.dimension or num_workers < 30:
            self.optims = [CMA(instrumentation, budget=None, num_workers=num_workers),
                           CMA(instrumentation, budget=None, num_workers=num_workers)]
            self.budget_before_choosing = budget // 3
        for optim in self.optims:
            optim._random_state = self.random_state  # share the state


@registry.register
class CM(CMandAS2):
    """Competence map, simplest."""

    def __init__(self, instrumentation: Union[int, Instrumentation], budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        assert budget is not None
        self.optims = [TwoPointsDE(instrumentation, budget=None, num_workers=num_workers)]  # noqa: F405
        self.budget_before_choosing = 2 * budget
        if budget < 201:
            self.optims = [OnePlusOne(instrumentation, budget=None, num_workers=num_workers)]
        if budget > 50 * self.dimension:
            self.optims = [CMA(instrumentation, budget=None, num_workers=num_workers)]
        for optim in self.optims:
            optim._random_state = self.random_state  # share the state


@registry.register
class MultiCMA(CM):
    """Combining 3 CMAs. Exactly identical. Active selection at 1/10 of the budget."""

    def __init__(self, instrumentation: Union[int, Instrumentation], budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        assert budget is not None
        self.optims = [CMA(instrumentation, budget=None, num_workers=num_workers),
                       CMA(instrumentation, budget=None, num_workers=num_workers),
                       CMA(instrumentation, budget=None, num_workers=num_workers)]
        for optim in self.optims:
            optim._random_state = self.random_state  # share the state
        self.budget_before_choosing = budget // 10


@registry.register
class TripleCMA(CM):
    """Combining 3 CMAs. Exactly identical. Active selection at 1/3 of the budget."""

    def __init__(self, instrumentation: Union[int, Instrumentation], budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        assert budget is not None
        self.optims = [CMA(instrumentation, budget=None, num_workers=num_workers),
                       CMA(instrumentation, budget=None, num_workers=num_workers),
                       CMA(instrumentation, budget=None, num_workers=num_workers)]
        for optim in self.optims:
            optim._random_state = self.random_state  # share the state
        self.budget_before_choosing = budget // 3


@registry.register
class MultiScaleCMA(CM):
    """Combining 3 CMAs with different init scale. Active selection at 1/3 of the budget."""

    def __init__(self, instrumentation: Union[int, Instrumentation], budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        self.optims = [CMA(instrumentation, budget=None, num_workers=num_workers),
                       MilliCMA(instrumentation, budget=None, num_workers=num_workers),
                       MicroCMA(instrumentation, budget=None, num_workers=num_workers)]
        for optim in self.optims:
            optim._random_state = self.random_state  # share the state
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
        x = [kwargs[f'x{i}'] for i in range(len(kwargs))]
        xr, value = self._registered[0]
        if not np.array_equal(x, xr):
            raise ValueError("Call does not match registered")
        self._registered.clear()
        return value


class _BO(base.Optimizer):

    def __init__(self, instrumentation: Union[int, Instrumentation], budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        self._parameters = ParametrizedBO()
        self._transform = transforms.ArctanBound(0, 1)
        self._bo: Optional[BayesianOptimization] = None
        self._fake_function = _FakeFunction()

    @property
    def bo(self) -> BayesianOptimization:
        if self._bo is None:
            params = self._parameters
            bounds = {f'x{i}': (0., 1.) for i in range(self.dimension)}
            self._bo = BayesianOptimization(self._fake_function, bounds, random_state=self.random_state)
            if self._parameters.gp_parameters is not None:
                self._bo.set_gp_params(**self._parameters.gp_parameters)
            # init
            init = params.initialization
            if params.middle_point:
                self._bo.probe([.5] * self.dimension, lazy=True)
            elif init is None:
                self._bo._queue.add(self._bo._space.random_sample())
            if init is not None:
                init_budget = int(np.sqrt(self.budget) if params.init_budget is None else params.init_budget)
                init_budget -= params.middle_point
                if init_budget > 0:
                    sampler = {"Hammersley": sequences.HammersleySampler,
                               "LHS": sequences.LHSSampler,
                               "random": sequences.RandomSampler}[init](self.dimension, budget=init_budget,
                                                                        scrambling=(init == "Hammersley"),
                                                                        random_state=self.random_state)
                    for point in sampler:
                        self._bo.probe(point, lazy=True)
        return self._bo

    def _internal_ask_candidate(self) -> base.Candidate:
        p = self._parameters
        util = UtilityFunction(kind=p.utility_kind, kappa=p.utility_kappa, xi=p.utility_xi)
        try:
            x_probe = next(self.bo._queue)
        except StopIteration:
            x_probe = self.bo.suggest(util)  # this is time consuming
            x_probe = [x_probe[f'x{i}'] for i in range(len(x_probe))]
        data = self._transform.backward(np.array(x_probe, copy=False))
        candidate = self.create_candidate.from_data(data)
        candidate._meta["x_probe"] = x_probe
        return candidate

    def _internal_tell_candidate(self, candidate: base.Candidate, value: float) -> None:
        if "x_probe" in candidate._meta:
            y = candidate._meta["x_probe"]
        else:
            y = self._transform.forward(np.array(candidate.data, copy=False))  # tell not asked
        self._fake_function.register(y, -value)  # minimizing
        self.bo.probe(y, lazy=False)
        # for some unknown reasons, BO wants to evaluate twice the same point,
        # but since it keeps a cache of the values, the registered value is not used
        # so we should clean the "fake" function
        self._fake_function._registered.clear()

    def _internal_provide_recommendation(self) -> base.ArrayLike:
        return self._transform.backward(np.array([self.bo.max['params'][f'x{i}'] for i in range(self.dimension)]))


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

    def __init__(self, *,
                 initialization: Optional[str] = None,
                 init_budget: Optional[int] = None,
                 middle_point: bool = False,
                 utility_kind: str = "ucb",  # bayes_opt default
                 utility_kappa: float = 2.576,
                 utility_xi: float = 0.0,
                 gp_parameters: Optional[Dict[str, Any]] = None) -> None:
        assert initialization is None or initialization in ["random", "Hammersley", "LHS"], f'Unknown init {initialization}'
        self.initialization = initialization
        self.init_budget = init_budget
        self.middle_point = middle_point
        self.utility_kind = utility_kind
        self.utility_kappa = utility_kappa
        self.utility_xi = utility_xi
        self.gp_parameters = gp_parameters
        super().__init__()

    def __call__(self, instrumentation: Union[int, Instrumentation],
                 budget: Optional[int] = None, num_workers: int = 1) -> base.Optimizer:
        gp_params = {} if self.gp_parameters is None else self.gp_parameters
        if isinstance(instrumentation, Instrumentation) and gp_params.get("alpha", 0) == 0:
            noisy = instrumentation.noisy
            cont = instrumentation.continuous
            if noisy or not cont:
                warnings.warn("Dis-continuous and noisy instrumentation require gp_parameters['alpha'] > 0 "
                              "(for your instrumentation, continuity={cont} and noisy={noisy}).\n"
                              "Find more information on BayesianOptimization's github.\n"
                              "You should then create a new instance of optimizerlib.ParametrizedBO with appropriate parametrization.",
                              InefficientSettingsWarning)
        return super().__call__(instrumentation, budget, num_workers)


BO = ParametrizedBO().with_name("BO", register=True)
RBO = ParametrizedBO(initialization="random").with_name("RBO", register=True)
QRBO = ParametrizedBO(initialization="Hammersley").with_name("QRBO", register=True)
MidQRBO = ParametrizedBO(initialization="Hammersley", middle_point=True).with_name("MidQRBO", register=True)
LBO = ParametrizedBO(initialization="LHS").with_name("LBO", register=True)


@registry.register
class EMNA_pure(base.Optimizer):
    # pylint: disable=too-many-instance-attributes

    def __init__(self, instrumentation: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        try:
          instrumentation = instrumentation.dimension
        except:
          instrumentation = int(instrumentation)
        self.instrumentation = instrumentation
        self.sigma = np.ones(instrumentation)
        self.mu = instrumentation
        self.llambda = 4 * instrumentation
        self.previousBest = None
        self.secondToLastBest = None

        if num_workers is not None:
            self.llambda = max(self.llambda, num_workers)
        self.largeenough = False
        if num_workers/instrumentation > 8:
            self.largeenough = True
        self.current_center = np.zeros(instrumentation)
        # Evaluated population
        self.evaluated_population: List[base.ArrayLike] = []
        self.evaluated_population_sigma: List[float] = []
        self.evaluated_population_fitness: List[float] = []
        # Unevaluated population
        self.unevaluated_population: List[base.ArrayLike] = []
        self.unevaluated_population_sigma: List[float] = []

    def _internal_provide_recommendation(self) -> base.ArrayLike:
        return self.current_bests["optimistic"].x

    def _internal_ask(self) -> base.ArrayLike:
        # mutated_sigma = self.sigma * np.exp(np.random.normal(0, 1) / np.sqrt(self.instrumentation))
        mutated_sigma = self.sigma * np.random.normal(0, 1, self.instrumentation)
        individual = tuple(self.current_center + mutated_sigma)
        self.unevaluated_population_sigma += [mutated_sigma]
        self.unevaluated_population += [tuple(individual)]
        return individual

    def _internal_tell(self, x: base.ArrayLike, value: float) -> None:
        idx = self.unevaluated_population.index(tuple(x))
        self.evaluated_population += [x]
        self.evaluated_population_fitness += [value]
        self.evaluated_population_sigma += [self.unevaluated_population_sigma[idx]]
        del self.unevaluated_population[idx]
        del self.unevaluated_population_sigma[idx]
        if len(self.evaluated_population) >= self.llambda:
            # Sorting the population.
            sorted_pop_with_sigma_and_fitness = [(i, s, f) for f, i, s in sorted(
                zip(self.evaluated_population_fitness, self.evaluated_population, self.evaluated_population_sigma))]
            self.evaluated_population = [p[0] for p in sorted_pop_with_sigma_and_fitness]
            self.evaluated_population_sigma = [p[1] for p in sorted_pop_with_sigma_and_fitness]
            self.evaluated_population_fitness = [p[2] for p in sorted_pop_with_sigma_and_fitness]
            # Computing the new parent.
            # EMNA update
            self.current_center = sum([np.asarray(self.evaluated_population[i]) for i in range(self.mu)]) / self.mu
            t1 = [(self.evaluated_population[i]-self.current_center)**2 for i in range(self.mu)]
            self.sigma = np.sqrt(sum(t1)/(self.mu))
            self.evaluated_population = []
            self.evaluated_population_sigma = []
            self.evaluated_population_fitness = []


@registry.register
class EMNA_loglambda(base.Optimizer):
    # pylint: disable=too-many-instance-attributes

    def __init__(self, instrumentation: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        try:
          instrumentation = instrumentation.dimension
        except:
          instrumentation = int(instrumentation)
        self.instrumentation = instrumentation
        self.sigma = np.ones(instrumentation)
        self.mu = instrumentation
        self.llambda = 4 * instrumentation
        self.previousBest = None
        self.secondToLastBest = None

        if num_workers is not None:
            self.llambda = max(self.llambda, num_workers)
        self.largeenough = False
        if num_workers/instrumentation > 8:
            self.largeenough = True
        self.current_center = np.zeros(instrumentation)
        # Evaluated population
        self.evaluated_population: List[base.ArrayLike] = []
        self.evaluated_population_sigma: List[float] = []
        self.evaluated_population_fitness: List[float] = []
        # Unevaluated population
        self.unevaluated_population: List[base.ArrayLike] = []
        self.unevaluated_population_sigma: List[float] = []

    def _internal_provide_recommendation(self) -> base.ArrayLike:
        return self.current_bests["optimistic"].x

    def _internal_ask(self) -> base.ArrayLike:
        # mutated_sigma = self.sigma * np.exp(np.random.normal(0, 1) / np.sqrt(self.instrumentation))
        mutated_sigma = self.sigma * np.random.normal(0, 1, self.instrumentation)
        individual = tuple(self.current_center + mutated_sigma)
        self.unevaluated_population_sigma += [mutated_sigma]
        self.unevaluated_population += [tuple(individual)]
        return individual

    def _internal_tell(self, x: base.ArrayLike, value: float) -> None:
        idx = self.unevaluated_population.index(tuple(x))
        self.evaluated_population += [x]
        self.evaluated_population_fitness += [value]
        self.evaluated_population_sigma += [self.unevaluated_population_sigma[idx]]
        del self.unevaluated_population[idx]
        del self.unevaluated_population_sigma[idx]
        if len(self.evaluated_population) >= self.llambda:
            # Sorting the population.
            sorted_pop_with_sigma_and_fitness = [(i, s, f) for f, i, s in sorted(
                zip(self.evaluated_population_fitness, self.evaluated_population, self.evaluated_population_sigma))]
            self.evaluated_population = [p[0] for p in sorted_pop_with_sigma_and_fitness]
            self.evaluated_population_sigma = [p[1] for p in sorted_pop_with_sigma_and_fitness]
            self.evaluated_population_fitness = [p[2] for p in sorted_pop_with_sigma_and_fitness]
            # Computing the new parent.
            # EMNA update
            self.current_center = sum([np.asarray(self.evaluated_population[i]) for i in range(self.mu)]) / self.mu
            t1 = [(self.evaluated_population[i]-self.current_center)**2 for i in range(self.mu)]
            if self.largeenough:
                self.sigma /=  max(1,(np.log(self.llambda)/2)**(1/(self.instrumentation)))
            self.sigma = np.sqrt(sum(t1)/(self.mu))
            self.evaluated_population = []
            self.evaluated_population_sigma = []
            self.evaluated_population_fitness = []

@registry.register
class EMNA_aniso(base.Optimizer):
    # pylint: disable=too-many-instance-attributes

    def __init__(self, instrumentation: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        instrumentation = instrumentation.dimension
        try:
          instrumentation = instrumentation.dimension
        except:
          instrumentation = int(instrumentation)
        self.instrumentation = instrumentation
        self.sigma = np.ones(instrumentation)
        self.mu = instrumentation
        self.llambda = 4 * instrumentation
        self.previousBest = None
        self.secondToLastBest = None

        if num_workers is not None:
            self.llambda = max(self.llambda, num_workers)
        self.largeenough = False
        if num_workers/instrumentation > 8:
            self.largeenough = True
        self.current_center = np.zeros(instrumentation)
        # Evaluated population
        self.evaluated_population: List[base.ArrayLike] = []
        self.evaluated_population_sigma: List[float] = []
        self.evaluated_population_fitness: List[float] = []
        # Unevaluated population
        self.unevaluated_population: List[base.ArrayLike] = []
        self.unevaluated_population_sigma: List[float] = []
        # Archive
        self.archive_fitness: List[float] = []

    def _internal_provide_recommendation(self) -> base.ArrayLike:
        return self.current_bests["optimistic"].x

    def _internal_ask(self) -> base.ArrayLike:
        # mutated_sigma = self.sigma * np.exp(np.random.normal(0, 1) / np.sqrt(self.instrumentation))
        mutated_sigma = self.sigma * np.random.normal(0, 1, self.instrumentation)
        individual = tuple(self.current_center + mutated_sigma)
        self.unevaluated_population_sigma += [mutated_sigma]
        self.unevaluated_population += [tuple(individual)]
        return individual

    def _internal_tell(self, x: base.ArrayLike, value: float) -> None:
        self.archive_fitness += [value]
        if (not self.largeenough):
            if len(self.archive_fitness) >= 5 * self.llambda:
                first_fifth = [self.archive_fitness[i] for i in range(self.llambda)]
                last_fifth = [self.archive_fitness[i] for i in range(4*self.llambda, 5*self.llambda)]
                mean1 = sum(first_fifth) / float(self.llambda)
                std1 = np.std(first_fifth) / np.sqrt(self.llambda - 1)
                mean2 = sum(last_fifth) / float(self.llambda)
                std2 = np.std(last_fifth) / np.sqrt(self.llambda - 1)
                z = (mean1 - mean2) / (np.sqrt(std1**2 + std2**2))
                if z < 2.:
                    self.mu *= 2
                else:
                    self.mu = int(self.mu * 0.84)
                    if self.mu < self.instrumentation:
                        self.mu = self.instrumentation
                self.llambda = 4 * self.mu
                if self.num_workers > 1:
                    self.llambda = max(self.llambda, self.num_workers)
                    self.mu = self.llambda // 4
                self.archive_fitness = []
        idx = self.unevaluated_population.index(tuple(x))
        self.evaluated_population += [x]
        self.evaluated_population_fitness += [value]
        self.evaluated_population_sigma += [self.unevaluated_population_sigma[idx]]
        del self.unevaluated_population[idx]
        del self.unevaluated_population_sigma[idx]
        if len(self.evaluated_population) >= self.llambda:
            # Sorting the population.
            sorted_pop_with_sigma_and_fitness = [(i, s, f) for f, i, s in sorted(
                zip(self.evaluated_population_fitness, self.evaluated_population, self.evaluated_population_sigma))]
            self.evaluated_population = [p[0] for p in sorted_pop_with_sigma_and_fitness]
            self.evaluated_population_sigma = [p[1] for p in sorted_pop_with_sigma_and_fitness]
            self.evaluated_population_fitness = [p[2] for p in sorted_pop_with_sigma_and_fitness]
            # Computing the new parent.
            # EMNA update
            self.current_center = sum([np.asarray(self.evaluated_population[i]) for i in range(self.mu)]) / self.mu
            t1 = [(self.evaluated_population[i]-self.current_center)**2 for i in range(self.mu)]
            self.sigma = np.sqrt(sum(t1)/(self.mu))
            self.evaluated_population = []
            self.evaluated_population_sigma = []
            self.evaluated_population_fitness = []


@registry.register
class EMNA_mm(base.Optimizer):
    # pylint: disable=too-many-instance-attributes

    def __init__(self, instrumentation: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        try:
          instrumentation = instrumentation.dimension
        except:
          instrumentation = int(instrumentation)
        self.instrumentation = instrumentation
        self.sigma = 1
        self.mu = instrumentation
        self.llambda = 4 * instrumentation
        self.previousBest = None
        self.secondToLastBest = None

        if num_workers is not None:
            self.llambda = max(self.llambda, num_workers)
        self.largeenough = False
        if num_workers/instrumentation > 8:
            self.largeenough = True
        self.current_center = np.zeros(instrumentation)
        # Evaluated population
        self.evaluated_population: List[base.ArrayLike] = []
        self.evaluated_population_sigma: List[float] = []
        self.evaluated_population_fitness: List[float] = []
        # Unevaluated population
        self.unevaluated_population: List[base.ArrayLike] = []
        self.unevaluated_population_sigma: List[float] = []

    def _internal_provide_recommendation(self) -> base.ArrayLike:
        return self.current_bests["optimistic"].x

    def _internal_ask(self) -> base.ArrayLike:
        mutated_sigma = self.sigma * np.exp(np.random.normal(0, 1) / np.sqrt(self.instrumentation))
        individual = tuple(self.current_center + mutated_sigma * np.random.normal(0, 1, self.instrumentation))
        self.unevaluated_population_sigma += [mutated_sigma]
        self.unevaluated_population += [tuple(individual)]
        return individual

    def _internal_tell(self, x: base.ArrayLike, value: float) -> None:
        idx = self.unevaluated_population.index(tuple(x))
        self.evaluated_population += [x]
        self.evaluated_population_fitness += [value]
        self.evaluated_population_sigma += [self.unevaluated_population_sigma[idx]]
        del self.unevaluated_population[idx]
        del self.unevaluated_population_sigma[idx]
        if len(self.evaluated_population) >= self.llambda:
            # Sorting the population.
            sorted_pop_with_sigma_and_fitness = [(i, s, f) for f, i, s in sorted(
                zip(self.evaluated_population_fitness, self.evaluated_population, self.evaluated_population_sigma))]
            self.evaluated_population = [p[0] for p in sorted_pop_with_sigma_and_fitness]
            self.evaluated_population_sigma = [p[1] for p in sorted_pop_with_sigma_and_fitness]
            self.evaluated_population_fitness = [p[2] for p in sorted_pop_with_sigma_and_fitness]
            # Computing the new parent.
            # EMNA update
            self.current_center = sum([np.asarray(self.evaluated_population[i]) for i in range(self.mu)]) / self.mu
            t1 = [(self.evaluated_population[i]-self.current_center)**2 for i in range(self.mu)]
            self.sigma = np.sqrt(sum(t1)/(self.mu))
            self.evaluated_population = []
            self.evaluated_population_sigma = []
            self.evaluated_population_fitness = []


@registry.register
class EMNA_win(base.Optimizer):
    # pylint: disable=too-many-instance-attributes

    def __init__(self, instrumentation: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        try:
          instrumentation = instrumentation.dimension
        except:
          instrumentation = int(instrumentation)
        self.instrumentation = instrumentation
        self.sigma = 1
        self.mu = instrumentation
        self.llambda = 4 * instrumentation
        self.previousBest = None
        self.secondToLastBest = None

        if num_workers is not None:
            self.llambda = max(self.llambda, num_workers)
        self.largeenough = False
        if num_workers/instrumentation > 8:
            self.largeenough = True
        self.current_center = np.zeros(instrumentation)
        # Evaluated population
        self.evaluated_population: List[base.ArrayLike] = []
        self.evaluated_population_sigma: List[float] = []
        self.evaluated_population_fitness: List[float] = []
        # Unevaluated population
        self.unevaluated_population: List[base.ArrayLike] = []
        self.unevaluated_population_sigma: List[float] = []
        # Archive
        self.archive_fitness: List[float] = []

    def _internal_provide_recommendation(self) -> base.ArrayLike:
        return self.current_bests["optimistic"].x

    def _internal_ask(self) -> base.ArrayLike:
        mutated_sigma = self.sigma * np.exp(np.random.normal(0, 1) / np.sqrt(self.instrumentation))
        individual = tuple(self.current_center + mutated_sigma * np.random.normal(0, 1, self.instrumentation))
        self.unevaluated_population_sigma += [mutated_sigma]
        self.unevaluated_population += [tuple(individual)]
        return individual

    def _internal_tell(self, x: base.ArrayLike, value: float) -> None:
        self.archive_fitness += [value]
        if (not self.largeenough):
            if len(self.archive_fitness) >= 5 * self.llambda:
                first_fifth = [self.archive_fitness[i] for i in range(self.llambda)]
                last_fifth = [self.archive_fitness[i] for i in range(4*self.llambda, 5*self.llambda)]
                mean1 = sum(first_fifth) / float(self.llambda)
                std1 = np.std(first_fifth) / np.sqrt(self.llambda - 1)
                mean2 = sum(last_fifth) / float(self.llambda)
                std2 = np.std(last_fifth) / np.sqrt(self.llambda - 1)
                z = (mean1 - mean2) / (np.sqrt(std1**2 + std2**2))
                if z < 2.:
                    self.mu *= 2
                else:
                    self.mu = int(self.mu * 0.84)
                    if self.mu < self.instrumentation:
                        self.mu = self.instrumentation
                self.llambda = 4 * self.mu
                if self.num_workers > 1:
                    self.llambda = max(self.llambda, self.num_workers)
                    self.mu = self.llambda // 4
                self.archive_fitness = []
        idx = self.unevaluated_population.index(tuple(x))
        self.evaluated_population += [x]
        self.evaluated_population_fitness += [value]
        self.evaluated_population_sigma += [self.unevaluated_population_sigma[idx]]
        del self.unevaluated_population[idx]
        del self.unevaluated_population_sigma[idx]
        if len(self.evaluated_population) >= self.llambda:
            # Sorting the population.
            sorted_pop_with_sigma_and_fitness = [(i, s, f) for f, i, s in sorted(
                zip(self.evaluated_population_fitness, self.evaluated_population, self.evaluated_population_sigma))]
            self.evaluated_population = [p[0] for p in sorted_pop_with_sigma_and_fitness]
            self.evaluated_population_sigma = [p[1] for p in sorted_pop_with_sigma_and_fitness]
            self.evaluated_population_fitness = [p[2] for p in sorted_pop_with_sigma_and_fitness]
            # Computing the new parent.
            # EMNA update
            self.current_center = sum([np.asarray(self.evaluated_population[i]) for i in range(self.mu)]) / self.mu
            t1 = [(self.evaluated_population[i]-self.current_center)**2 for i in range(self.mu)]
            self.sigma = np.sqrt(sum(t1)/(self.mu))
            self.evaluated_population = []
            self.evaluated_population_sigma = []
            self.evaluated_population_fitness = []



@registry.register
class EMNA_aniso_win(base.Optimizer):
    # pylint: disable=too-many-instance-attributes

    def __init__(self, instrumentation: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        try:
          instrumentation = instrumentation.dimension
        except:
          instrumentation = int(instrumentation)
        self.instrumentation = instrumentation
        self.sigma = np.ones(instrumentation)
        self.mu = instrumentation
        self.llambda = 4 * instrumentation
        self.previousBest = None
        self.secondToLastBest = None

        if num_workers is not None:
            self.llambda = max(self.llambda, num_workers)
        self.largeenough = False
        if num_workers/instrumentation > 8:
            self.largeenough = True
        self.current_center = np.zeros(instrumentation)
        # Evaluated population
        self.evaluated_population: List[base.ArrayLike] = []
        self.evaluated_population_sigma: List[float] = []
        self.evaluated_population_fitness: List[float] = []
        # Unevaluated population
        self.unevaluated_population: List[base.ArrayLike] = []
        self.unevaluated_population_sigma: List[float] = []
        # Archive
        self.archive_fitness: List[float] = []

    def _internal_provide_recommendation(self) -> base.ArrayLike:
        return self.current_bests["optimistic"].x

    def _internal_ask(self) -> base.ArrayLike:
        mutated_sigma = self.sigma * np.exp(np.random.normal(0, 1, self.instrumentation) / np.sqrt(self.instrumentation))
        individual = tuple(self.current_center + mutated_sigma * np.random.normal(0, 1, self.instrumentation))
        self.unevaluated_population_sigma += [mutated_sigma]
        self.unevaluated_population += [tuple(individual)]
        return individual

    def _internal_tell(self, x: base.ArrayLike, value: float) -> None:
        self.archive_fitness += [value]
        if (not self.largeenough):
            if len(self.archive_fitness) >= 5 * self.llambda:
                first_fifth = [self.archive_fitness[i] for i in range(self.llambda)]
                last_fifth = [self.archive_fitness[i] for i in range(4*self.llambda, 5*self.llambda)]
                mean1 = sum(first_fifth) / float(self.llambda)
                std1 = np.std(first_fifth) / np.sqrt(self.llambda - 1)
                mean2 = sum(last_fifth) / float(self.llambda)
                std2 = np.std(last_fifth) / np.sqrt(self.llambda - 1)
                z = (mean1 - mean2) / (np.sqrt(std1**2 + std2**2))
                if z < 2.:
                    self.mu *= 2
                else:
                    self.mu = int(self.mu * 0.84)
                    if self.mu < self.instrumentation:
                        self.mu = self.instrumentation
                self.llambda = 4 * self.mu
                if self.num_workers > 1:
                    self.llambda = max(self.llambda, self.num_workers)
                    self.mu = self.llambda // 4
                self.archive_fitness = []
        idx = self.unevaluated_population.index(tuple(x))
        self.evaluated_population += [x]
        self.evaluated_population_fitness += [value]
        self.evaluated_population_sigma += [self.unevaluated_population_sigma[idx]]
        del self.unevaluated_population[idx]
        del self.unevaluated_population_sigma[idx]
        if len(self.evaluated_population) >= self.llambda:
            # Sorting the population.
            sorted_pop_with_sigma_and_fitness = [(i, s, f) for f, i, s in sorted(
                zip(self.evaluated_population_fitness, self.evaluated_population, self.evaluated_population_sigma))]
            self.evaluated_population = [p[0] for p in sorted_pop_with_sigma_and_fitness]
            self.evaluated_population_sigma = [p[1] for p in sorted_pop_with_sigma_and_fitness]
            self.evaluated_population_fitness = [p[2] for p in sorted_pop_with_sigma_and_fitness]
            # Computing the new parent.
            # EMNA update
            self.current_center = sum([np.asarray(self.evaluated_population[i]) for i in range(self.mu)]) / self.mu
            t1 = [(self.evaluated_population[i]-self.current_center)**2 for i in range(self.mu)]
            self.sigma = np.sqrt(sum(t1)/(self.mu))
            self.evaluated_population = []
            self.evaluated_population_sigma = []
            self.evaluated_population_fitness = []

@registry.register
class EMNA_iso(base.Optimizer):
    # pylint: disable=too-many-instance-attributes

    def __init__(self, instrumentation: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        try:
          instrumentation = instrumentation.dimension
        except:
          instrumentation = int(instrumentation)
        self.instrumentation = instrumentation
        self.sigma = 1
        self.mu = instrumentation
        self.llambda = 4 * instrumentation
        self.previousBest = None
        self.secondToLastBest = None

        if num_workers is not None:
            self.llambda = max(self.llambda, num_workers)
        self.largeenough = False
        if num_workers/instrumentation > 8:
            self.largeenough = True
        self.current_center = np.zeros(instrumentation)
        # Evaluated population
        self.evaluated_population: List[base.ArrayLike] = []
        self.evaluated_population_sigma: List[float] = []
        self.evaluated_population_fitness: List[float] = []
        # Unevaluated population
        self.unevaluated_population: List[base.ArrayLike] = []
        self.unevaluated_population_sigma: List[float] = []

    def _internal_provide_recommendation(self) -> base.ArrayLike:
        return self.current_bests["optimistic"].x

    def _internal_ask(self) -> base.ArrayLike:
        mutated_sigma = self.sigma
        individual = tuple(self.current_center + mutated_sigma * np.random.normal(0, 1, self.instrumentation))
        self.unevaluated_population_sigma += [mutated_sigma]
        self.unevaluated_population += [tuple(individual)]
        return individual

    def _internal_tell(self, x: base.ArrayLike, value: float) -> None:
        idx = self.unevaluated_population.index(tuple(x))
        self.evaluated_population += [x]
        self.evaluated_population_fitness += [value]
        self.evaluated_population_sigma += [self.unevaluated_population_sigma[idx]]
        del self.unevaluated_population[idx]
        del self.unevaluated_population_sigma[idx]
        if len(self.evaluated_population) >= self.llambda:
            # Sorting the population.
            sorted_pop_with_sigma_and_fitness = [(i, s, f) for f, i, s in sorted(
                zip(self.evaluated_population_fitness, self.evaluated_population, self.evaluated_population_sigma))]
            self.evaluated_population = [p[0] for p in sorted_pop_with_sigma_and_fitness]
            self.evaluated_population_sigma = [p[1] for p in sorted_pop_with_sigma_and_fitness]
            self.evaluated_population_fitness = [p[2] for p in sorted_pop_with_sigma_and_fitness]
            # Computing the new parent.
            # EMNA update
            self.current_center = sum([np.asarray(self.evaluated_population[i]) for i in range(self.mu)]) / self.mu
            t1 = [(self.evaluated_population[i]-self.current_center)**2 for i in range(self.mu)]
            self.sigma = np.sqrt(sum(t1)/(self.mu))
            self.evaluated_population = []
            self.evaluated_population_sigma = []
            self.evaluated_population_fitness = []
__all__ = list(registry.keys())

