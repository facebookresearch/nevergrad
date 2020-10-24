# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import nevergrad.common.typing as tp
# import numpy as np
from nevergrad.parametrization import parameter as p
from nevergrad.optimization.utils import UidQueue
from . import base
from .evolution_ops import rankers as rankers


class _EvolutionStrategy(base.Optimizer):
    """Experimental evolution-strategy-like algorithm
    The behavior is going to evolve
    """

    def __init__(
        self,
        parametrization: base.IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        *,
        config: tp.Optional["EvolutionStrategy"] = None
    ) -> None:
        if budget is not None and budget < 60:
            warnings.warn("ES algorithms are inefficient with budget < 60", base.InefficientSettingsWarning)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._population: tp.Dict[str, p.Parameter] = {}
        self._uid_queue = UidQueue()
        self._waiting: tp.List[p.Parameter] = []
        # configuration
        self._config = EvolutionStrategy() if config is None else config
        self._ranker = None
        if self._config.selection == "simple":
            self._internal_tell_es = self._simple_tell_candidate
        elif self._config.selection == "nsga2":
            self._ranker = rankers.FastNonDominatedRanking()
            self._internal_tell_es = self._nsga2_tell_candidate
        else:
            raise NotImplementedError


    def _internal_ask_candidate(self) -> p.Parameter:
        if self.num_ask < self._config.popsize:
            param = self.parametrization.sample()
            assert param.uid == param.heritage["lineage"]  # this is an assumption used below
            self._uid_queue.asked.add(param.uid)
            self._population[param.uid] = param
            return param
        uid = self._uid_queue.ask()
        param = self._population[uid].spawn_child()
        param.mutate()
        ratio = self._config.recombination_ratio
        if ratio and self._rng.rand() < ratio:
            selected = self._rng.choice(list(self._population))
            param.recombine(self._population[selected])
        return param


    def _internal_tell_candidate(self, candidate: p.Parameter, value: tp.FloatLoss) -> None:
        self._internal_tell_es(candidate, value)


    def _simple_tell_candidate(self, candidate: p.Parameter, value: tp.FloatLoss) -> None:
        candidate._meta["value"] = value
        if self._config.offsprings is None:
            uid = candidate.heritage["lineage"]
            self._uid_queue.tell(uid)
            parent_value = float('inf') if uid not in self._population else self._population[uid]._meta["value"]
            if value < parent_value:
                self._population[uid] = candidate
        else:
            if candidate.parents_uids[0] not in self._population and len(self._population) < self._config.popsize:
                self._population[candidate.uid] = candidate
                self._uid_queue.tell(candidate.uid)
            else:
                self._waiting.append(candidate)
            if len(self._waiting) >= self._config.offsprings:
                self._simple_selection()


    def _simple_selection(self):
        choices = self._waiting + ([] if self._config.only_offsprings else list(self._population.values()))
        choices.sort(key=lambda x: x._meta["value"])
        self._population = {x.uid: x for x in choices[:self._config.popsize]}
        self._uid_queue.clear()
        self._waiting.clear()
        for uid in self._population:
            self._uid_queue.tell(uid)


    def _nsga2_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss): #tp.FloatLoss?
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
        self._nsga2_selection()
        # clean up
        self._uid_queue.clear()
        self._waiting.clear()
        for uid in self._population:
            self._uid_queue.tell(uid)


    def _nsga2_selection(self):
        # Refer to section C of the NSGA paper
        popsize = len(self._population)
        joined_population = dict(self._population)
        for candidate in self._waiting: #_waiting stores offsprings
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
        self._population = population_next
        


class EvolutionStrategy(base.ConfiguredOptimizer):
    """Experimental evolution-strategy-like algorithm
    The API is going to evolve
    """

    # pylint: disable=unused-argument
    def __init__(
            self,
            *,
            recombination_ratio: float = 0,
            popsize: int = 40,
            offsprings: tp.Optional[int] = None,
            only_offsprings: bool = False,
            # de_step: bool = False,
            selection: str = "simple"
    ) -> None:
        super().__init__(_EvolutionStrategy, locals(), as_config=True)
        assert offsprings is None or not only_offsprings or offsprings > popsize
        if only_offsprings:
            assert offsprings is not None, "only_offsprings only work if offsprings is not None (non-DE mode)"
        assert 0 <= recombination_ratio <= 1
        assert selection in ["simple", "nsga2"]
        self.recombination_ratio = recombination_ratio
        self.popsize = popsize
        self.offsprings = offsprings
        self.only_offsprings = only_offsprings
        self.selection = selection


RecES = EvolutionStrategy(recombination_ratio=1, only_offsprings=True, offsprings=60).set_name("RecES", register=True)
RecMixES = EvolutionStrategy(recombination_ratio=1, only_offsprings=False, offsprings=20).set_name("RecMixES", register=True)
RecMutDE = EvolutionStrategy(recombination_ratio=1, only_offsprings=False, offsprings=None).set_name("RecMutDE", register=True)
ES = EvolutionStrategy(recombination_ratio=0, only_offsprings=True, offsprings=60).set_name("ES", register=True)
MixES = EvolutionStrategy(recombination_ratio=0, only_offsprings=False, offsprings=20).set_name("MixES", register=True)
MutDE = EvolutionStrategy(recombination_ratio=0, only_offsprings=False, offsprings=None).set_name("MutDE", register=True)
NSGAIIES = EvolutionStrategy(recombination_ratio=0, only_offsprings=True, offsprings=60, selection="nsga2").set_name("NSGAIIES", register=True)
