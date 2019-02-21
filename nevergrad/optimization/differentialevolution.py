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


@base.registry.register
class DE(base.Optimizer):
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
        self._lhs_init = False
        self._qr_init = False
        self._por_DE = False
        self._recommendation = "optimistic"
        self.llambda = max(30, num_workers)
        self.scale = 1.0
        self.population: List[Optional[ArrayLike]] = []
        self.candidates: List[Optional[ArrayLike]] = []
        self.population_fitnesses: List[Optional[float]] = []
        self.inoculation = False
        self.hyperinoc = False
        self.sampler: Optional[sequences.Sampler] = None
        self.NF = False  # This is not a noise-free variant of DE.
        # parameters
        self.CR = 0.5
        self.F1 = 0.8
        self.F2 = 0.8
        self.k = 0  # crossover

    def match_population_size_to_lambda(self) -> None:
        # TODO: Ideally, this should be done only once in __init__ and/or eventually when changing the value of
        # self.llambda
        if len(self.population) < self.llambda:
            self.candidates += [None] * (self.llambda - len(self.population))
            self.population_fitnesses += [None] * (self.llambda - len(self.population))
            self.population += [None] * (self.llambda - len(self.population))

    def _internal_provide_recommendation(self) -> Tuple[float, ...]:  # This is NOT the naive version. We deal with noise.
        if self._recommendation == "noisy":
            med_fitness = np.median([f for f in self.population_fitnesses if f is not None])
            good_guys = [p for p, f in zip(self.population, self.population_fitnesses) if f is not None and f < med_fitness]
            if not good_guys:
                return self.current_bests["pessimistic"].x
            return sum([np.array(g) for g in good_guys]) / len(good_guys)  # type: ignore
        else:
            return self.current_bests[self._recommendation].x

    def _internal_ask(self) -> Tuple[float, ...]:
        if self._lhs_init and self.sampler is None:
            self.sampler = sequences.LHSSampler(self.dimension, budget=self.llambda)
        if self._qr_init and self.sampler is None:
            self.sampler = sequences.ScrHammersleySampler(self.dimension, budget=self.llambda)
        self.match_population_size_to_lambda()
        location = self._num_ask % self.llambda
        i = (self.population[location])
        a, b, c = (self.population[np.random.randint(self.llambda)] for _ in range(3))

        if self._por_DE:
            self.CR = np.random.uniform(0., 1.)

        if any(x is None for x in [i, a, b, c]):
            if self.inoculation:
                inoc = float(location) / float(self.llambda)
            else:
                inoc = 1.
            if self.hyperinoc:
                p = [float(self.llambda - location), location]
                p = [p_ / sum(p) for p_ in p]
                sample = self.sampler() if (self._lhs_init or self._qr_init) else np.random.normal(0, 1, self.dimension)  # type: ignore
                new_guy = tuple([np.random.choice([0, self.scale * sample[i]], p=p) for i in range(self.dimension)])
            else:
                new_guy = tuple(inoc * self.scale * (np.random.normal(0, 1, self.dimension)
                                                     if not (self._qr_init or self._lhs_init)
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
        if self.hashed:
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
            donor = i + self.F1 * (a - b) + self.F2 * (self.current_bests["pessimistic"].x - i)
        k = self.k
        assert k <= 2
        if k == 0 or self.dimension < 3:
            R = np.random.randint(self.dimension)
            for idx in range(self.dimension):
                if idx != R and np.random.uniform(0, 1) > self.CR:
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


class DifferentialEvolution(base.OptimizerFamily):

    # pylint: disable=unused-argument,too-many-arguments
    def __init__(self, lhs_init: bool = False, qr_init: bool = False, por_DE: bool = False, scale: Union[str, float] = 1.,
                 inoculation: bool = False, hyperinoc: bool = False, recommendation: str = "optimistic",
                 CR: float = .5, F1: float = .8, F2: float = .8, crossover: int = 0):
        super().__init__(**{x: y for x, y in locals().items() if x not in ["self", "__class__"]})
        assert recommendation in ["optimistic", "pessimistic", "noisy", "mean"]

    def __call__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> DE:
        run = DE(dimension=dimension, budget=budget, num_workers=num_workers)
        # ugly but effective :s
        for name, value in self._kwargs.items():
            rename = name
            if hasattr(run, "_" + name):
                rename = "_" + rename
            elif name == "crossover":
                rename = "k"
            elif name == "scale" and isinstance(value, str):
                if value == "mini":
                    value = 1. / np.sqrt(dimension)
                else:
                    raise ValueError(f'Unknown scaling: "{value}".')
            setattr(run, rename, value)
        run.name = repr(self)
        return run


OnePointDE = DifferentialEvolution(crossover=1).with_name("OnePointDE", register=True)
TwoPointsDE = DifferentialEvolution(crossover=2).with_name("TwoPointsDE", register=True)
LhsDE = DifferentialEvolution(lhs_init=True).with_name("LhsDE", register=True)
QrDE = DifferentialEvolution(qr_init=True).with_name("QrDE", register=True)
MiniDE = DifferentialEvolution(scale="mini").with_name("MiniDE", register=True)
MiniLhsDE = DifferentialEvolution(lhs_init=True, scale="mini").with_name("MiniLhsDE", register=True)
MiniQrDE = DifferentialEvolution(qr_init=True, scale="mini").with_name("MiniQrDE", register=True)
OnePointDE = DifferentialEvolution(recommendation="noisy").with_name("NoisyDE", register=True)
AlmostRotationInvariantDE = DifferentialEvolution(CR=.9).with_name("AlmostRotationInvariantDE", register=True)


@base.registry.register
class AlmostRotationInvariantDEAndBigPop(DE):

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.CR = 0.9
        self.llambda = max(self.llambda, dimension+1)


@base.registry.register
class RotationInvariantDE(DE):

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.CR = 1
        self.llambda = max(self.llambda, dimension+1)


@base.registry.register
class BPRotationInvariantDE(DE):

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.CR = 1
        self.llambda = max(self.llambda, 7*dimension)
