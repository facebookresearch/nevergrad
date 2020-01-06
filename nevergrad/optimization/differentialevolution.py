# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union, Set
import warnings
import numpy as np
from scipy import stats
from . import base
from .base import IntOrParameter
from . import sequences


class Crossover:

    def __init__(self, random_state: np.random.RandomState, crossover: Union[str, float]):
        self.CR = .5
        self.crossover = crossover
        self.random_state = random_state
        if isinstance(crossover, float):
            self.CR = crossover
        elif crossover == "random":
            self.CR = self.random_state.uniform(0., 1.)
        elif crossover not in ["twopoints", "onepoint"]:
            raise ValueError(f'Unknown crossover "{crossover}"')

    def apply(self, donor: np.ndarray, individual: np.ndarray) -> None:
        dim = donor.size
        if self.crossover == "twopoints" and dim >= 4:
            return self.twopoints(donor, individual)
        elif self.crossover == "onepoint" and dim >= 3:
            return self.onepoint(donor, individual)
        else:
            return self.variablewise(donor, individual)

    def variablewise(self, donor: np.ndarray, individual: np.ndarray) -> None:
        R = self.random_state.randint(donor.size)
        # the following could be updated to vectorial uniform sampling (changes recomms)
        transfer = np.array([idx != R and self.random_state.uniform(0, 1) > self.CR for idx in range(donor.size)])
        donor[transfer] = individual[transfer]

    def onepoint(self, donor: np.ndarray, individual: np.ndarray) -> None:
        R = self.random_state.randint(1, donor.size)
        if self.random_state.choice([True, False]):
            donor[R:] = individual[R:]
        else:
            donor[:R] = individual[:R]

    def twopoints(self, donor: np.ndarray, individual: np.ndarray) -> None:
        bounds = sorted(self.random_state.choice(donor.size + 1, size=2, replace=False).tolist())
        if bounds[1] == donor.size and not bounds[0]:  # make sure there is at least one point crossover
            bounds[self.random_state.randint(2)] = self.random_state.randint(1, donor.size)
        if self.random_state.choice([True, False]):
            donor[bounds[0]: bounds[1]] = individual[bounds[0]: bounds[1]]
        else:
            donor[:bounds[0]] = individual[:bounds[0]]
            donor[bounds[1]:] = individual[bounds[1]:]


class _DE(base.Optimizer):
    """Differential evolution.

    Default pop size equal to 30
    We return the mean of the individuals with fitness better than median, which might be stupid sometimes.
    CR =.5, F1=.8, F2=.8, curr-to-best.
    Initial population: pure random.
    """
    # pylint: disable=too-many-locals, too-many-nested-blocks
    # pylint: disable=too-many-branches, too-many-statements

    def __init__(self, instrumentation: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        self._penalize_cheap_violations = True
        self._parameters = DifferentialEvolution()
        self._llambda: Optional[int] = None
        self.population: base.utils.Population[base.utils.Individual] = base.utils.Population([])
        self.sampler: Optional[sequences.Sampler] = None
        self._replaced: Set[bytes] = set()

    @property
    def scale(self) -> float:
        scale = self._parameters.scale
        if isinstance(scale, str):
            assert scale == "mini"  # computing on demand because it requires to know the dimension
            return float(1. / np.sqrt(self.dimension))
        return scale

    @property
    def llambda(self) -> int:
        if self._llambda is None:  # computing on demand because it requires to know the dimension
            pop_choice = {"standard": 0, "dimension": self.dimension + 1, "large": 7 * self.dimension}
            self._llambda = max(30, self.num_workers, pop_choice[self._parameters.popsize])
        return self._llambda

    def _internal_provide_recommendation(self) -> np.ndarray:  # This is NOT the naive version. We deal with noise.
        if self._parameters.recommendation != "noisy":
            return self.current_bests[self._parameters.recommendation].x
        med_fitness = np.median([p.value for p in self.population if p.value is not None])
        good_guys = [p for p in self.population if p.value is not None and p.x is not None and p.value < med_fitness]
        if not good_guys:
            return self.current_bests["pessimistic"].x
        return sum([g.x for g in good_guys]) / len(good_guys)  # type: ignore

    def _internal_ask_candidate(self) -> base.Candidate:
        if len(self.population) < self.llambda:  # initialization phase
            init = self._parameters.initialization
            if self.sampler is None and init != "gaussian":
                assert init in ["LHS", "QR"]
                sampler_cls = sequences.LHSSampler if init == "LHS" else sequences.HammersleySampler
                self.sampler = sampler_cls(self.dimension, budget=self.llambda, scrambling=init == "QR", random_state=self._rng)
            new_guy = self.scale * (self._rng.normal(0, 1, self.dimension)
                                    if self.sampler is None else stats.norm.ppf(self.sampler()))
            particle = base.utils.Individual(new_guy)
            self.population.extend([particle])
            self.population.get_queued(remove=True)  # since it was just added
            candidate = self.create_candidate.from_data(new_guy)
            candidate._meta["particle"] = particle
            return candidate
        # init is done
        particle = self.population.get_queued(remove=True)
        individual = particle.x
        # define donor
        indiv_a, indiv_b = (self.population[self.population.uids[self._rng.randint(self.llambda)]].x for _ in range(2))
        assert indiv_a is not None and indiv_b is not None
        donor = (individual + self._parameters.F1 * (indiv_a - indiv_b) +
                 self._parameters.F2 * (self.current_bests["pessimistic"].x - individual))
        # apply crossover
        co = self._parameters.crossover
        crossovers = Crossover(self._rng, 1. / self.dimension if co == "dimension" else co)
        crossovers.apply(donor, individual)
        # create candidate
        candidate = self.create_candidate.from_data(donor)
        candidate._meta["particle"] = particle
        return candidate

    def _internal_tell_candidate(self, candidate: base.Candidate, value: float) -> None:
        particle: base.utils.Individual = candidate._meta["particle"]  # all asked candidate should have this field
        if not particle._active:
            self._internal_tell_not_asked(candidate, value)
            return
        if particle.value is None or value <= particle.value:
            particle.x = candidate.data
            particle.value = value
        self.population.set_queued(particle)

    def _internal_tell_not_asked(self, candidate: base.Candidate, value: float) -> None:
        worst_part = None
        if not len(self.population) < self.llambda:
            worst_part = max(iter(self.population), key=lambda p: p.value if p.value is not None else np.inf)
            if worst_part.value is not None and worst_part.value < value:
                return  # no need to update
        particle = base.utils.Individual(candidate.data)
        particle.value = value
        if worst_part is not None:
            self.population.replace(worst_part, particle)
            worst_part._active = False


# pylint: disable=too-many-arguments, too-many-instance-attributes
class DifferentialEvolution(base.ParametrizedFamily):
    """Differential evolution algorithms.

    Default pop size is 30
    We return the mean of the individuals with fitness better than median, which might be stupid sometimes.
    Default settings are CR =.5, F1=.8, F2=.8, curr-to-best.
    Initial population: pure random.

    Parameters
    ----------
    initialization: "LHS", "QR" or "gaussian"
        algorithm/distribution used for the initialization phase
    scale: float or str
        scale of random component of the updates
    recommendation: "pessimistic", "optimistic", "mean" or "noisy"
        choice of the criterion for the best point to recommend
    crossover: float or str
        crossover rate value, or strategy among:
        - "dimension": crossover rate of  1 / dimension,
        - "random": different random (uniform) crossover rate at each iteration
        - "onepoint": one point crossover
        - "twopoints": two points crossover
    F1: float
        differential weight #1
    F2: float
        differential weight #2
    popsize: "standard", "dimension", "large"
        size of the population to use. "standard" is max(num_workers, 30), "dimension" max(num_workers, 30, dimension +1)
        and "large" max(num_workers, 30, 7 * dimension).
    """

    _optimizer_class = _DE

    def __init__(self, *, initialization: str = "gaussian", scale: Union[str, float] = 1.,
                 recommendation: str = "optimistic", crossover: Union[str, float] = .5,
                 F1: float = .8, F2: float = .8, popsize: str = "standard") -> None:
        # initial checks
        assert recommendation in ["optimistic", "pessimistic", "noisy", "mean"]
        assert initialization in ["gaussian", "LHS", "QR"]
        assert isinstance(scale, float) or scale == "mini"
        assert popsize in ["large", "dimension", "standard"]
        assert isinstance(crossover, float) or crossover in ["onepoint", "twopoints", "dimension", "random"]
        self.initialization = initialization
        self.scale = scale
        self.recommendation = recommendation
        # parameters
        self.F1 = F1
        self.F2 = F2
        self.crossover = crossover
        self.popsize = popsize
        super().__init__()

    def __call__(self, instrumentation: IntOrParameter,
                 budget: Optional[int] = None, num_workers: int = 1) -> base.Optimizer:
        if budget is not None and budget < 60:
            warnings.warn("DE algorithms are inefficient with budget < 60", base.InefficientSettingsWarning)
        return super().__call__(instrumentation, budget, num_workers)


DE = DifferentialEvolution().with_name("DE", register=True)
OnePointDE = DifferentialEvolution(crossover="onepoint").with_name("OnePointDE", register=True)
TwoPointsDE = DifferentialEvolution(crossover="twopoints").with_name("TwoPointsDE", register=True)
LhsDE = DifferentialEvolution(initialization="LHS").with_name("LhsDE", register=True)
QrDE = DifferentialEvolution(initialization="QR").with_name("QrDE", register=True)
MiniDE = DifferentialEvolution(scale="mini").with_name("MiniDE", register=True)
MiniLhsDE = DifferentialEvolution(initialization="LHS", scale="mini").with_name("MiniLhsDE", register=True)
MiniQrDE = DifferentialEvolution(initialization="QR", scale="mini").with_name("MiniQrDE", register=True)
NoisyDE = DifferentialEvolution(recommendation="noisy").with_name("NoisyDE", register=True)
AlmostRotationInvariantDE = DifferentialEvolution(crossover=.9).with_name("AlmostRotationInvariantDE", register=True)
AlmostRotationInvariantDEAndBigPop = DifferentialEvolution(crossover=.9, popsize="dimension").with_name(
    "AlmostRotationInvariantDEAndBigPop", register=True)
RotationInvariantDE = DifferentialEvolution(crossover=1., popsize="dimension").with_name("RotationInvariantDE", register=True)
BPRotationInvariantDE = DifferentialEvolution(crossover=1., popsize="large").with_name("BPRotationInvariantDE", register=True)
