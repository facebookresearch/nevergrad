# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, List, Dict, Tuple
from collections import defaultdict
import numpy as np
from scipy import stats
import cma
from . import base
from . import mutations
from .base import registry
# families of optimizers
# pylint: disable=unused-wildcard-import,wildcard-import
from .differentialevolution import *
from .oneshot import *
from .recastlib import *


# # # # # optimizers # # # # #


@registry.register
class OnePlusOne(base.Optimizer):
    """Simple but sometimes powerful optimization algorithm.

    We use the one-fifth adaptation rule, going back to Schumer and Steiglitz (1968).
    It was independently rediscovered by Devroye (1972) and Rechenberg (1973).
    We use asynchronous updates, so that the 1+1 can actually be parallel and even
    performs quite well in such a context - this is naturally close to 1+lambda.
    """

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.sigma: float = 1

    def _internal_ask(self) -> base.ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        else:
            return self.current_bests["pessimistic"].x + self.sigma * np.random.normal(0, 1, self.dimension)

    def _internal_tell(self, x: base.ArrayLike, value: float) -> None:
        if value <= self.current_bests["pessimistic"].mean:
            self.sigma = 2. * self.sigma
        else:
            self.sigma = .84 * self.sigma


@registry.register
class CauchyOnePlusOne(OnePlusOne):
    """Version of the OnePlusOne optimization algorithm with Cauchy mutations.

    Many papers use Cauchy mutations, maybe the first one was
    X. Yao, Y. Liu and G. Lin, Evolutionary Programing Made Faster, IEEE
    Transactions on Evolutionary Computation, vol. 3, 82-102, July 1999.
    """

    def _internal_ask(self) -> base.ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        else:
            return self.current_bests["pessimistic"].x + self.sigma * np.random.standard_cauchy(self.dimension)


@registry.register
class CMA(base.Optimizer):
    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.es = cma.CMAEvolutionStrategy([0.] * dimension, 1.0)
        self.listx: List[base.ArrayLike] = []
        self.listy: List[float] = []

    def _internal_ask(self) -> base.ArrayLike:
        return self.es.ask(1)[0]

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
        return self.es.result.xbest


@registry.register
class TBPSA(base.Optimizer):
    """Test-based population-size adaptation.

    Population-size equal to lambda = 4 x dimension.
    Test by comparing the first fifth and the last fifth of the 5lambda evaluations.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.sigma = 1
        self.mu = dimension
        self.llambda = 4 * dimension
        if num_workers is not None:
            self.llambda = max(self.llambda, num_workers)
        self.current_center = np.zeros(dimension)
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
        mutated_sigma = self.sigma * np.exp(np.random.normal(0, 1) / np.sqrt(self.dimension))
        individual = tuple(self.current_center + mutated_sigma * np.random.normal(0, 1, self.dimension))
        self.unevaluated_population_sigma += [mutated_sigma]
        self.unevaluated_population += [tuple(individual)]
        return individual

    def _internal_tell(self, x: base.ArrayLike, value: float) -> None:
        self.archive_fitness += [value]
        if len(self.archive_fitness) >= 5 * self.llambda:
            first_fifth = [self.archive_fitness[i] for i in range(self.llambda)]
            last_fifth = [self.archive_fitness[i] for i in range(self.llambda)]
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
            self.evaluated_population_sigma = [p[1] for p in sorted_pop_with_sigma_and_fitness]
            self.evaluated_population_fitness = [p[2] for p in sorted_pop_with_sigma_and_fitness]
            # Computing the new parent.
            self.current_center = sum([np.asarray(self.evaluated_population[i]) for i in range(self.mu)]) / self.mu
            self.sigma = np.exp(sum([np.log(self.evaluated_population_sigma[i]) for i in range(self.mu)]) / self.mu)
            self.evaluated_population = []
            self.evaluated_population_sigma = []
            self.evaluated_population_fitness = []


@registry.register
class NaiveTBPSA(TBPSA):

    def _internal_provide_recommendation(self) -> base.ArrayLike:
        return self.current_bests["optimistic"].x


@registry.register
class NoisyBandit(base.Optimizer):
    """UCB.

    This is upper confidence bound (adapted to minimization),
      with very poor parametrization; in particular, the logarithmic term is set to zero.
    Infinite arms: we add one arm when #trials >= #arms ** 3."""

    def _internal_ask(self) -> base.ArrayLike:
        if self._num_suggestions >= len(self.archive) ** 3:
            return np.random.normal(0, 1, self.dimension)
        if np.random.choice([True, False]):
            # numpy does not accept choice on list of tuples, must choose index instead
            idx = np.random.choice(len(self.archive))
            return list(self.archive.keys())[idx]
        return self.current_bests["optimistic"].x


@registry.register
class OptimisticDiscreteOnePlusOne(base.Optimizer):
    """Close to UCB, but new arms are chosen by discrete mutations from the best.

    This combines the discrete 1+1 algorithm and bandits."""

    def _internal_ask(self) -> base.ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        if self._num_suggestions <= len(self.archive) ** 3:
            return self.current_bests["optimistic"].x
        return mutations.discrete_mutation(self.current_bests["pessimistic"].x)


@registry.register
class RecombiningOptimisticNoisyDiscreteOnePlusOne(base.Optimizer):
    """Combining the discrete 1+1, noise management a la bandit, and genetic crossovers.

    Close to UCB, but new arms are chosen by discrete mutations from the current best and
    we crossover with the best every two new arms.
    This combines the discrete 1+1, bandits and genetic crossovers.
    """

    def _internal_ask(self) -> base.ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        elif self._num_suggestions <= len(self.archive) ** 3:
            return self.current_bests["optimistic"].x
        elif self._num_suggestions % 2 == 0 or len(self.archive) < 3:
            return mutations.discrete_mutation(self.current_bests["pessimistic"].x)
        else:
            return mutations.crossover(self.current_bests["pessimistic"].x,
                                       mutations.get_roulette(self.archive, num=2))


@registry.register
class DoubleFastGADiscreteOnePlusOne(base.Optimizer):
    """Close to discrete 1+1, but new arms are chosen by double-FastGA mutations from the current best.
    Doerr et al, Fast Genetic Algorithms, 2017
    """

    def _internal_ask(self) -> base.ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        return mutations.doubledoerr_discrete_mutation(self.current_bests["pessimistic"].x)


@registry.register
class FastGAOptimisticDiscreteOnePlusOne(base.Optimizer):
    """Close to discrete 1+1, but new arms are chosen by FastGA mutations from the current best.

    This is close to DoubleFastGA variants, but assumes that each variable has 2 possible values."""

    def _internal_ask(self) -> base.ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        return mutations.doerr_discrete_mutation(self.current_bests["pessimistic"].x)


@registry.register
class DoubleFastGAOptimisticNoisyDiscreteOnePlusOne(base.Optimizer):
    """Close to UCB and discrete 1+1, but new arms are chosen by double-FastGA mutations from the current best.
    Doerr et al, Fast Genetic Algorithms, 2017
    """

    def _internal_ask(self) -> base.ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        if self._num_suggestions <= len(self.archive) ** 3:
            return self.current_bests["optimistic"].x
        return mutations.doubledoerr_discrete_mutation(self.current_bests["pessimistic"].x)


@registry.register
class FastGAOptimisticNoisyDiscreteOnePlusOne(base.Optimizer):
    """Close to UCB, but new arms are chosen by FastGA mutations from the current best.

    This is close to DoubleFastGA variants, but assumes that each variable has 2 possible values."""

    def _internal_ask(self) -> base.ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        if self._num_suggestions <= len(self.archive) ** 3:
            return self.current_bests["optimistic"].x
        return mutations.doerr_discrete_mutation(self.current_bests["pessimistic"].x)


@registry.register
class PortfolioOptimisticNoisyDiscreteOnePlusOne(base.Optimizer):
    """Random number of mutated bits + bandit noise management + discrete 1+1 algorithm.

    The random number of bits is called uniform mixing in Dang & Lehre "Self-adaptation of Mutation Rates
    in Non-elitist Population", 2016."""

    def _internal_ask(self) -> base.ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        if 20 * self._num_suggestions <= len(self.archive) ** 3:
            return self.current_bests["optimistic"].x
        return mutations.portfolio_discrete_mutation(self.current_bests["pessimistic"].x)


@registry.register
class RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne(base.Optimizer):
    """Adding crossover to PortfolioOptimisticNoisyDiscreteOnePlusOneOptimizer."""

    def _internal_ask(self) -> base.ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        if 20 * self._num_suggestions <= len(self.archive) ** 3:
            return self.current_bests["optimistic"].x
        if self._num_suggestions % 2 == 0 or len(self.archive) < 3:
            return mutations.portfolio_discrete_mutation(self.current_bests["pessimistic"].x)
        else:
            return mutations.crossover(self.current_bests["pessimistic"].x,
                                       mutations.get_roulette(self.archive, num=2))


@registry.register
class NoisyDiscreteOnePlusOne(base.Optimizer):
    """Bandit + discrete mutations from current LCB-best."""

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)

    def _internal_ask(self) -> base.ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        if self._num_suggestions <= len(self.archive) ** 3:
            # numpy does not accept choice on list of tuples, must choose index instead
            idx = np.random.choice(len(self.archive))
            return list(self.archive.keys())[idx]
        return mutations.discrete_mutation(self.current_bests["pessimistic"].x)


@registry.register
class DiscreteOnePlusOne(base.Optimizer):
    """Discrete 1+1 optimization algorithm."""

    def _internal_ask(self) -> base.ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        return mutations.discrete_mutation(self.current_bests["pessimistic"].x)


@registry.register
class PortfolioDiscreteOnePlusOne(base.Optimizer):
    """Discrete 1+1 optimization algorithm with random number of mutated bits.

    The random number of bits is called uniform mixing in Dang & Lehre "Self-adaptation of Mutation Rates
    in Non-elitist Population", 2016.
    """

    def _internal_ask(self) -> base.ArrayLike:
        if not self._num_suggestions:
            return np.zeros(self.dimension)
        return mutations.portfolio_discrete_mutation(self.current_bests["pessimistic"].x)


@registry.register
class PSO(base.Optimizer):
    """Partially following SPSO2011. However, no randomization of the population order.
    """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.llambda = max(40, num_workers)
        self.pop: List[base.ArrayLike] = []
        self.pop_speed: List[base.ArrayLike] = []
        self.pop_best: List[base.ArrayLike] = []
        self.pop_best_fitness: List[Optional[float]] = []
        self.pop_fitness: List[Optional[float]] = []
        self.pso_best: Optional[base.ArrayLike] = None
        self.pso_best_fitness = float("inf")
        self.locations: Dict[Tuple[float, ...], List[int]] = defaultdict(list)
        self.index = -1
        self.omega = 0.5 / np.log(2.)
        self.phip = 0.5 + np.log(2.)
        self.phig = 0.5 + np.log(2.)
        self.eps = 1e-10

    def _internal_ask(self) -> base.ArrayLike:
        self.index += 1
        if self.index == 0:
            self.pso_best = None
            self.pso_best_fitness = float("inf")
            for i in range(self.llambda):
                guy = np.random.uniform(0., 1., self.dimension)
                self.pop += [guy]
                self.pop_best += [guy]
                self.pop_speed += [np.random.uniform(-1., 1., self.dimension)]
                self.pop_best_fitness += [float("inf")]
                self.pop_fitness += [None]
        # Focusing on the right guy in the population.
        location = self.index % self.llambda
        # First, the initialization.
        if self.pop_fitness[location] is None:  # This guy is not evaluated.
            assert self.pop[location] is not None
            guy = tuple(self.to_real(self.pop[location]))
            self.locations[guy] += [location]
            return guy
        # We are in a standard case.
        # Speed mutation.
        for i in range(self.dimension):
            rp = np.random.uniform(0., 1.)
            rg = np.random.uniform(0., 1.)
            self.pop_speed[location][i] = (  # type: ignore
                self.omega * self.pop_speed[location][i]
                + self.phip * rp * (self.pop_best[location][i]-self.pop[location][i])
                + self.phig * rg * (self.pso_best[i] - self.pop[location][i])  # type: ignore
            )
        # Particle mutation.
        self.pop[location] += self.pop_speed[location]
        self.pop[location] = [max(0.+self.eps, min(1.-self.eps, x_)) for x_ in self.pop[location]]
        guy = tuple(self.to_real(self.pop[location]))
        self.locations[guy] += [location]
        return guy

    def _internal_provide_recommendation(self) -> base.ArrayLike:
        return tuple(self.to_real(self.pso_best))

    def _internal_tell(self, x: base.ArrayLike, value: float) -> None:
        x = tuple(x)
        assert self.locations[x]
        location = self.locations[x][0]
        del self.locations[x][0]
        point = tuple(self.to_real(self.pop[location]))
        assert x == point, str(x) + f"{x} vs {point}     {self.pop}"
        self.pop_fitness[location] = value
        if value < self.pso_best_fitness:
            assert max(self.pop[location]) < 1., str(self.pop[location])
            assert min(self.pop[location]) > 0., str(self.pop[location])
            self.pso_best = [s for s in self.pop[location]]
            self.pso_best_fitness = value
        if value < self.pop_best_fitness[location]:  # type: ignore
            self.pop_best[location] = [s for s in self.pop[location]]
            self.pop_best_fitness[location] = value

    @staticmethod
    def to_real(x: base.ArrayLike) -> base.ArrayLike:
        output = stats.norm.ppf(x)
        assert not any(x for x in np.isnan(output)), f"Encountered NaN value {output}"
        return output


@registry.register
class SPSA(base.Optimizer):
    # pylint: disable=too-many-instance-attributes
    ''' The First order SPSA algorithm as shown in [1,2,3], with implementation details
    from [4,5].

    [1] https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation
    [2] https://www.chessprogramming.org/SPSA
    [3] Spall, James C. "Multivariate stochastic approximation using a simultaneous perturbation gradient approximation."
        IEEE transactions on automatic control 37.3 (1992): 332-341.
    [4] Section 7.5.2 in "Introduction to Stochastic Search and Optimization: Estimation, Simulation and Control" by James C. Spall.
    [5] Pushpendre Rastogi, Jingyi Zhu, James C. Spall CISS (2016).
        Efficient implementation of Enhanced Adaptive Simultaneous Perturbation Algorithms.
    '''
    no_parallelization = True

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self._rng = np.random.RandomState(np.random.randint(2**32))
        self.init = True
        self.idx = 0
        self.delta = float('nan')
        self.ym: Optional[np.ndarray] = None
        self.yp: Optional[np.ndarray] = None
        self.t = np.zeros(self.dimension)
        self.avg = np.zeros(self.dimension)
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

    def ck(self, k: int) -> float:
        'c_k determines the pertubation.'
        return self.c / (k//2 + 1)**0.101

    def ak(self, k: int) -> float:
        'a_k is the learning rate.'
        return self.a / (k//2 + 1 + self.A)**0.602

    def _internal_ask(self) -> base.ArrayLike:
        k = self.idx
        if k % 2 == 0:
            if not self.init:
                assert self.yp is not None and self.ym is not None
                self.t -= (self.ak(k) * (self.yp - self.ym) / 2 / self.ck(k)) * self.delta
                self.avg += (self.t - self.avg) / (k // 2 + 1)
            self.delta = 2 * self._rng.randint(2, size=self.dimension) - 1
            return self.t - self.ck(k) * self.delta
        return self.t + self.ck(k) * self.delta

    def _internal_tell(self, x: base.ArrayLike, value: float) -> None:
        setattr(self, ('ym' if self.idx % 2 == 0 else 'yp'), np.array(value, copy=True))
        self.idx += 1
        if self.init and self.yp is not None and self.ym is not None:
            self.init = False

    def _internal_provide_recommendation(self) -> base.ArrayLike:
        return self.avg
