import warnings
import typing as tp
# import numpy as np
from nevergrad.parametrization import parameter as p
from nevergrad.common.tools import OrderedSet
from . import base


class _EvolutionStrategy(base.Optimizer):

    def __init__(self, instrumentation: base.IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None:
        if budget is not None and budget < 60:
            warnings.warn("DE algorithms are inefficient with budget < 60", base.InefficientSettingsWarning)
        super().__init__(instrumentation, budget=budget, num_workers=num_workers)
        self._parameters = EvolutionStrategy()
        self._population: tp.Dict[str, p.Parameter] = {}
        self._told_queue = tp.Deque[str]()
        self._asked_queue = OrderedSet[str]()
        self._waiting: tp.List[p.Parameter] = []

    def _internal_ask_candidate(self) -> p.Parameter:
        if self.num_ask < self._parameters.popsize or not self._population:
            param = self.instrumentation.sample()
            return param
        if self._told_queue:
            uid = self._told_queue.popleft()
        else:
            uid = next(iter(self._asked_queue))
        param = self._population[uid].spawn_child()
        param.mutate()
        # if self._parameters.de_step and len(self._population) > 1:
        #     sdata = [self._population[s].get_standardized_data(reference=self.instrumentation)
        #              for s in self._rng.choice(list(self._population), 2, replace=False)]
        #     F1, F2 = 0.8, 0.8
        #     data = param.get_standardized_data(reference=self.instrumentation)
        #     data += F2 * (self.current_bests["pessimistic"].x - data)
        #     #data += F1 * (sdata[1] - sdata[0])
        #     param.set_standardized_data(data, reference=self.instrumentation)
        if self._parameters.recombinations:
            selected = self._rng.choice(list(self._population), self._parameters.recombinations, replace=False)
            param.recombine(*(self._population[s] for s in selected))
        self._asked_queue.add(uid)
        return param

    def _internal_tell_candidate(self, candidate: p.Parameter, value: float) -> None:
        candidate._meta["value"] = value
        if self._parameters.offsprings is None:
            uid = candidate.heritage["lineage"]
            parent_value = float('inf') if uid not in self._population else self._population[uid]._meta["value"]
            if value < parent_value:
                self._population[uid] = candidate
            self._told_queue.append(uid)
        else:
            uid = candidate.uid
            if candidate.parents_uids[0] not in self._population and len(self._population) < self._parameters.popsize:
                self._population[uid] = candidate
                self._told_queue.append(uid)
            else:
                self._waiting.append(candidate)
            if len(self._waiting) >= self._parameters.offsprings:
                choices = self._waiting + ([] if self._parameters.only_offsprings else list(self._population.values()))
                choices.sort(key=lambda x: x._meta["value"])
                self._population = {x.uid: x for x in choices[:self._parameters.popsize]}
                self._told_queue.clear()
                self._asked_queue.clear()
                self._waiting.clear()
                self._told_queue.extend(list(self._population))


class EvolutionStrategy(base.ParametrizedFamily):

    _optimizer_class = _EvolutionStrategy

    def __init__(
            self,
            *,
            recombinations: int = 0,
            popsize: int = 40,
            offsprings: tp.Optional[int] = None,
            only_offsprings: bool = False,
            # de_step: bool = False,
    ) -> None:
        assert offsprings is None or not only_offsprings or offsprings > popsize
        if only_offsprings:
            assert offsprings is not None, "only_offsprings only work if offsprings is not None (non-DE mode)"
        self.recombinations = recombinations
        self.popsize = popsize
        self.offsprings = offsprings
        self.only_offsprings = only_offsprings
        # self.de_step = de_step
        super().__init__()


RecES = EvolutionStrategy(recombinations=1, only_offsprings=True, offsprings=60).with_name("RecES", register=True)
RecMixES = EvolutionStrategy(recombinations=1, only_offsprings=False, offsprings=20).with_name("RecMixES", register=True)
RecMutDE = EvolutionStrategy(recombinations=1, only_offsprings=False, offsprings=None).with_name("RecMutDE", register=True)
ES = EvolutionStrategy(recombinations=0, only_offsprings=True, offsprings=60).with_name("ES", register=True)
MixES = EvolutionStrategy(recombinations=0, only_offsprings=False, offsprings=20).with_name("MixES", register=True)
MutDE = EvolutionStrategy(recombinations=0, only_offsprings=False, offsprings=None).with_name("MutDE", register=True)
