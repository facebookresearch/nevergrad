# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple, List, Union
import numpy as np
from scipy import stats
from ..common.typetools import ArrayLike
from . import base
from . import sequences


class _DE(base.Optimizer):
    """Differential evolution.

    Default pop size equal to 30
    We return the mean of the individuals with fitness better than median, which might be stupid sometimes.
    CR =.5, F1=.8, F2=.8, curr-to-best.
    Initial population: pure random.
    """
    # pylint: disable=too-many-locals, too-many-nested-blocks, too-many-instance-attributes
    # pylint: disable=too-many-branches, too-many-statements

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self._parameters = DifferentialEvolution()
        self._llambda: Optional[int] = None
        self.population: List[Optional[ArrayLike]] = []
        self.candidates: List[Optional[ArrayLike]] = []
        self.population_fitnesses: List[Optional[float]] = []
        self.sampler: Optional[sequences.Sampler] = None
        self.NF = False  # This is not a noise-free variant of DE.

    @property
    def scale(self) -> float:
        scale = self._parameters.scale
        if isinstance(scale, str):
            assert scale == "mini"  # computing on demand because it requires to know the dimension
            scale = 1. / np.sqrt(self.dimension)
        assert isinstance(scale, float)
        return scale

    @property
    def llambda(self) -> int:
        if self._llambda is None:  # computing on demand because it requires to know the dimension
            pop_choice = {"standard": 0, "dimension": self.dimension + 1, "large": 7 * self.dimension}
            self._llambda = max(30, self.num_workers, pop_choice[self._parameters.popsize])
        return self._llambda

    def match_population_size_to_lambda(self) -> None:
        current_pop = len(self.population)
        if current_pop < self.llambda:
            self.candidates.extend([None] * (self.llambda - current_pop))
            self.population_fitnesses.extend([None] * (self.llambda - current_pop))
            self.population.extend([None] * (self.llambda - current_pop))

    def _internal_provide_recommendation(self) -> np.ndarray:  # This is NOT the naive version. We deal with noise.
        if self._parameters.recommendation != "noisy":
            return self.current_bests[self._parameters.recommendation].x
        med_fitness = np.median([f for f in self.population_fitnesses if f is not None])
        good_guys = [p for p, f in zip(self.population, self.population_fitnesses) if f is not None and f < med_fitness]
        if not good_guys:
            return self.current_bests["pessimistic"].x
        return sum([np.array(g) for g in good_guys]) / len(good_guys)

    def _internal_ask(self) -> Tuple[float, ...]:
        init = self._parameters.initialization
        if self.sampler is None and init is not None:
            assert init in ["LHS", "QR"]
            sampler_cls = sequences.LHSSampler if init == "LHS" else sequences.HammersleySampler
            self.sampler = sampler_cls(self.dimension, budget=self.llambda, scrambling=init == "QR")
        self.match_population_size_to_lambda()
        location = self._num_ask % self.llambda
        i = (self.population[location])
        a, b, c = (self.population[np.random.randint(self.llambda)] for _ in range(3))

        CR = 1. / self.dimension if isinstance(self._parameters.CR, str) else self._parameters.CR
        if self._parameters.por_DE:
            CR = np.random.uniform(0., 1.)

        if any(x is None for x in [i, a, b, c]):
            if self._parameters.inoculation:
                inoc = float(location) / float(self.llambda)
            else:
                inoc = 1.
            if self._parameters.hyperinoc:
                p = [float(self.llambda - location), location]
                p = [p_ / sum(p) for p_ in p]
                sample = self.sampler() if init is not None else np.random.normal(0, 1, self.dimension)  # type: ignore
                new_guy = tuple([np.random.choice([0, self.scale * sample[i]], p=p) for i in range(self.dimension)])
            else:
                new_guy = tuple(inoc * self.scale * (np.random.normal(0, 1, self.dimension)
                                                     if init is None
                                                     else stats.norm.ppf(self.sampler())))  # type: ignore
            self.population[location] = new_guy
            self.population_fitnesses[location] = None
            assert self.candidates[location] is None
            self.candidates[location] = tuple(new_guy)
            return new_guy
        i = np.array(i)
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        if self._parameters.hashed:
            k = np.random.randint(3)
            if k == 0:
                if self.NF:
                    donor = np.random.normal(0, 1, self.dimension)
                else:
                    donor = i
            if k == 1:
                donor = a
            if k == 2:
                donor = np.array(self.current_bests["pessimistic"].x)
        else:
            donor = i + self._parameters.F1 * (a - b) + self._parameters.F2 * (self.current_bests["pessimistic"].x - i)
        k = self._parameters.crossover
        assert k <= 2
        if k == 0 or self.dimension < 3:
            R = np.random.randint(self.dimension)
            for idx in range(self.dimension):
                if idx != R and np.random.uniform(0, 1) > CR:
                    donor[idx] = i[idx]
        elif k == 1 or self.dimension < 4:
            R = np.random.choice(np.arange(1, self.dimension))
            if np.random.uniform(0., 1.) < .5:
                for idx in range(R):
                    donor[idx] = i[idx]
            else:
                for idx in range(R, self.dimension):
                    donor[idx] = i[idx]
        elif k == 2:
            Ra, Rb = np.random.choice(self.dimension - 1, size=2, replace=False)
            if np.random.uniform(0., 1.) < .5:
                for idx in range(self.dimension):
                    if (idx - Ra) * (idx - Rb) >= 0:
                        donor[idx] = i[idx]
            else:
                for idx in range(self.dimension):
                    if (idx - Ra) * (idx - Rb) <= 0:
                        donor[idx] = i[idx]
        donor = tuple(donor)
        if self.candidates[location] is not None:
            for idx in range(self.llambda):
                if self.candidates[idx] is None:
                    location = idx
                    break
        assert self.candidates[location] is None
        self.candidates[location] = tuple(donor)
        return donor  # type: ignore

    def _internal_tell(self, x: ArrayLike, value: float) -> None:
        self.match_population_size_to_lambda()
        x = tuple(x)
        if x in self.candidates:
            idx = self.candidates.index(x)
        else:
            # If the point is not in candidates, either find an empty spot or choose randomly
            empty_indexes = [idx for idx, cand in enumerate(self.population) if cand is None]
            if empty_indexes:
                # We found an empty spot
                idx = empty_indexes[0]
            else:
                # No empty spot, choose randomly
                # TODO: There might be a more efficient approach than choosing at random
                idx = np.random.randint(len(self.candidates))
        if self.population_fitnesses[idx] is None or value <= self.population_fitnesses[idx]:  # type: ignore
            self.population[idx] = x
            self.population_fitnesses[idx] = value
        self.candidates[idx] = None

    def tell_not_asked(self, x: base.ArrayLike, value: float) -> None:
        raise base.TellNotAskedNotSupportedError


# pylint: disable=too-many-arguments, too-many-instance-attributes
class DifferentialEvolution(base.ParametrizedFamily):

    _optimizer_class = _DE

    def __init__(self, *, initialization: Optional[str] = None, por_DE: bool = False, scale: Union[str, float] = 1.,
                 inoculation: bool = False, hyperinoc: bool = False, recommendation: str = "optimistic", NF: bool = True,
                 CR: Union[str, float] = .5, F1: float = .8, F2: float = .8, crossover: int = 0, popsize: str = "standard",
                 hashed: bool = False) -> None:
        """Differential evolution algorithms.

        Default pop size is 30
        We return the mean of the individuals with fitness better than median, which might be stupid sometimes.
        Default settings are CR =.5, F1=.8, F2=.8, curr-to-best.
        Initial population: pure random.

        Parameters
        ----------
        initialization: "LHS", "QR" or None
            algorithm for the initialization phase
        por_DE: bool
            TODO
        scale: float
            scale of random component of the updates
        inoculation: bool
            TODO
        hyperinoc: bool
            TODO
        recommendation: "pessimistic", "optimistic", "mean" or "noisy"
            choice of the criterion for the best point to recommend
        CR: float
            TODO
        F1: float
            TODO
        F2: float
            TODO
        crossover: int
            TODO
        popsize: "standard", "dimension", "large"
            size of the population to use. "standard" is max(num_workers, 30), "dimension" max(num_workers, 30, dimension +1)
            and "large" max(num_workers, 30, 7 * dimension).
        NF: bool
            TODO
        hashed: bool
            TODO
        """
        # initial checks
        assert recommendation in ["optimistic", "pessimistic", "noisy", "mean"]
        assert crossover in [0, 1, 2]
        assert initialization in [None, "LHS", "QR"]
        assert isinstance(scale, float) or scale == "mini"
        assert popsize in ["large", "dimension", "standard"]
        assert isinstance(CR, float) or CR == "dimension"
        self.initialization = initialization
        self.por_DE = por_DE
        self.scale = scale
        self.inoculation = inoculation
        self.hyperinoc = hyperinoc
        self.recommendation = recommendation
        # parameters
        self.CR = CR
        self.F1 = F1
        self.F2 = F2
        self.crossover = crossover
        self.popsize = popsize
        self.NF = NF
        self.hashed = hashed
        super().__init__()


DE = DifferentialEvolution().with_name("DE", register=True)
OnePointDE = DifferentialEvolution(crossover=1).with_name("OnePointDE", register=True)
TwoPointsDE = DifferentialEvolution(crossover=2).with_name("TwoPointsDE", register=True)
LhsDE = DifferentialEvolution(initialization="LHS").with_name("LhsDE", register=True)
QrDE = DifferentialEvolution(initialization="QR").with_name("QrDE", register=True)
MiniDE = DifferentialEvolution(scale="mini").with_name("MiniDE", register=True)
MiniLhsDE = DifferentialEvolution(initialization="LHS", scale="mini").with_name("MiniLhsDE", register=True)
MiniQrDE = DifferentialEvolution(initialization="QR", scale="mini").with_name("MiniQrDE", register=True)
NoisyDE = DifferentialEvolution(recommendation="noisy").with_name("NoisyDE", register=True)
AlmostRotationInvariantDE = DifferentialEvolution(CR=.9).with_name("AlmostRotationInvariantDE", register=True)
AlmostRotationInvariantDEAndBigPop = DifferentialEvolution(CR=.9, popsize="dimension").with_name("AlmostRotationInvariantDEAndBigPop",
                                                                                                 register=True)
RotationInvariantDE = DifferentialEvolution(CR=1., popsize="dimension").with_name("RotationInvariantDE", register=True)
BPRotationInvariantDE = DifferentialEvolution(CR=1., popsize="large").with_name("BPRotationInvariantDE", register=True)
