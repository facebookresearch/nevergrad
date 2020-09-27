
import warnings
import numpy as np
from scipy import stats
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from nevergrad.optimization.utils import UidQueue
from . import base
from .base import IntOrParameter
from . import sequences
#from . import differentialevolution
from .evolution_ops import rankers as rankers


from enum import Enum
from typing import TypeVar, List


class _NondominatedSortingGeneticAlgorithmII(base.Optimizer):
    """"NSGA-II
    """

    def __init__(
            self,
            parametrization: IntOrParameter,
            budget: tp.Optional[int] = None,
            num_workers: int = 1,
            config: tp.Optional["NSGA-II"] = None
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        # config
        self._config = NondominatedSortingGeneticAlgorithmII() if config is None else config
        self.scale = float(1. / np.sqrt(self.dimension)) if isinstance(self._config.scale, str) else self._config.scale
        pop_choice = {
            "standard": 0,
            "dimension": self.dimension + 1,
            "large": 7 * self.dimension
        }
        if isinstance(self._config.popsize, int):
            self.max_popsize = self._config.popsize
        else:
            self.max_popsize = max(30, self.num_workers, pop_choice[self._config.popsize])
        self._MULTIOBJECTIVE_AUTO_BOUND = max(self._MULTIOBJECTIVE_AUTO_BOUND, self.max_popsize)
        self._penalize_cheap_violations = True
        self._uid_queue = base.utils.UidQueue()
        self._waiting: tp.List[p.Parameter] = []
        self._population: tp.Dict[str, p.Parameter] = {}
        self._sampler: tp.Optional[sequences.Sampler] = None
        self._ranker = rankers.FastNonDominatedRanking()
        self._density_estimator = rankers.CrowdingDistance()


    def _internal_ask_candidate(self) -> p.Parameter:
        if len(self._population) < self.max_popsize: 
            # initialization phase
            init = self._config.initialization
            if self._sampler is None and init != "gaussian":
                assert init in ["LHS", "QR"]
                sampler_cls = sequences.LHSSampler if init == "LHS" else sequences.HammersleySampler
                self._sampler = sampler_cls(self.dimension, budget=self.max_popsize, scrambling=init == "QR", random_state=self._rng)
            new_guy = self.scale * (self._rng.normal(0, 1, self.dimension)
                                    if self._sampler is None else stats.norm.ppf(self._sampler()))
            candidate = self.parametrization.spawn_child().set_standardized_data(new_guy)
            candidate.heritage["lineage"] = candidate.uid  # new lineage
            candidate.loss = float("inf")
            self._population[candidate.uid] = candidate
            self._uid_queue.asked.add(candidate.uid)
            return candidate
        # init is done
        # propose next candidate of interest
        parent_uid = self._uid_queue.ask()
        candidate = self._population[parent_uid].spawn_child()
        ratio = self._config.mutation
        if ratio and self._rng.rand() < ratio:
            candidate.mutate()
        ratio = self._config.crossover
        if ratio and self._rng.rand() < ratio:
            selected = self._rng.choice(list(self._population))
            candidate.recombine(self._population[selected])
        return candidate


    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        uid = candidate.heritage["lineage"]
        #self._uid_queue.tell(uid) #request complete (asked -> told)
        if candidate.uid not in self._population and len(self._population) < self.max_popsize:
            self._population[candidate.uid] = candidate
            self._uid_queue.tell(candidate.uid) #next suggested candidate
            return
        if len(self._waiting) <= self.max_popsize: # number of offsprings and number of parents are equal
            self._waiting.append(candidate)
            return
        # Select candidates in the next generation
        self._population = self._selection_scheme(self._population, self._waiting)
        # clean up
        self._uid_queue.clear()
        self._waiting.clear()
        for uid in self._population:
            self._uid_queue.tell(uid)


    def _selection_scheme(self, population: tp.Dict[str, p.Parameter], offsprings = tp.List[p.Parameter]) -> tp.Dict[str, p.Parameter]:
        # Refer to section C of the NSGA paper
        popsize = len(population)
        joined_population = population
        for candidate in offsprings:
            joined_population[candidate.uid] = candidate
        fronts = self._ranker.compute_ranking(joined_population)
        #print(fronts)
        population_next: tp.Dict[str, p.Parameter] = {}
        count = 0
        for front_i in range(len(fronts)):
            count += len(fronts[front_i])
            if count >= popsize:
                self._density_estimator.compute_distance(fronts[front_i])
                self._density_estimator.sort(fronts[front_i])
                for c_i in range(0, popsize - len(population_next)):
                    population_next[fronts[front_i][c_i].uid] = fronts[front_i][c_i]
                break
            for candidate in fronts[front_i]:
                population_next[candidate.uid] = candidate
        return population_next


class NondominatedSortingGeneticAlgorithmII(base.ConfiguredOptimizer):
    def __init__(
            self,
            *,
            initialization: str = "gaussian",
            crossover: float = .5, #TODO: tp.Union[str, float] = .5,
            mutation: float = .5,
            popsize: tp.Union[str, int] = "standard",
            scale: tp.Union[str, float] = 1.,
    ) -> None:
        super().__init__(_NondominatedSortingGeneticAlgorithmII, locals(), as_config=True)
        assert initialization in ["gaussian", "LHS", "QR"]
        assert isinstance(crossover, float) #or crossover in ["onepoint", "twopoints", "dimension", "random", "parametrization"]
        assert isinstance(mutation, float)
        if not isinstance(popsize, int):
            assert popsize in ["large", "dimension", "standard"]
        assert isinstance(scale, float) or scale == "mini"

        self.initialization = initialization
        self.crossover = crossover
        self.mutation = mutation
        self.popsize = popsize
        self.scale = scale

NSGAII = NondominatedSortingGeneticAlgorithmII().set_name("NSGAII", register=True)
LhsNSGAII = NondominatedSortingGeneticAlgorithmII(initialization="LHS").set_name("LhsNSGAII", register=True)
QrNSGAII = NondominatedSortingGeneticAlgorithmII(initialization="QR").set_name("QrNSGAII", register=True)
