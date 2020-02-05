import warnings
import typing as tp
# import numpy as np
from nevergrad.parametrization import parameter as p
from nevergrad.optimization.utils import UidQueue
from . import base


class _EvolutionStrategy(base.Optimizer):
    """Experimental evolution-strategy-like algorithm
    The behavior is going to evolve
    """

    def __init__(self, parametrization: base.IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None:
        if budget is not None and budget < 60:
            warnings.warn("ES algorithms are inefficient with budget < 60", base.InefficientSettingsWarning)
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._parameters = EvolutionStrategy()
        self._population: tp.Dict[str, p.Parameter] = {}
        self._uid_queue = UidQueue()
        self._waiting: tp.List[p.Parameter] = []

    def _internal_ask_candidate(self) -> p.Parameter:
        if self.num_ask < self._parameters.popsize or not self._population:
            param = self.parametrization.sample()
            assert param.uid == param.heritage["lineage"]  # this is an assumption used below
            self._uid_queue.asked.add(param.uid)
            return param
        uid = self._uid_queue.ask()
        param = self._population[uid].spawn_child()
        param.mutate()
        ratio = self._parameters.recombination_ratio
        if ratio and self._rng.rand() < ratio:
            selected = self._rng.choice(list(self._population))
            param.recombine(self._population[selected])
        return param

    def _internal_tell_candidate(self, candidate: p.Parameter, value: float) -> None:
        candidate._meta["value"] = value
        if self._parameters.offsprings is None:
            uid = candidate.heritage["lineage"]
            self._uid_queue.tell(uid)
            parent_value = float('inf') if uid not in self._population else self._population[uid]._meta["value"]
            if value < parent_value:
                self._population[uid] = candidate
        else:
            if candidate.parents_uids[0] not in self._population and len(self._population) < self._parameters.popsize:
                self._population[candidate.uid] = candidate
                self._uid_queue.tell(candidate.uid)
            else:
                self._waiting.append(candidate)
            if len(self._waiting) >= self._parameters.offsprings:
                choices = self._waiting + ([] if self._parameters.only_offsprings else list(self._population.values()))
                choices.sort(key=lambda x: x._meta["value"])
                self._population = {x.uid: x for x in choices[:self._parameters.popsize]}
                self._uid_queue.clear()
                self._waiting.clear()
                for uid in self._population:
                    self._uid_queue.tell(uid)


class EvolutionStrategy(base.ParametrizedFamily):
    """Experimental evolution-strategy-like algorithm
    The API is going to evolve
    """

    _optimizer_class = _EvolutionStrategy

    def __init__(
            self,
            *,
            recombination_ratio: float = 0,
            popsize: int = 40,
            offsprings: tp.Optional[int] = None,
            only_offsprings: bool = False,
            # de_step: bool = False,
    ) -> None:
        assert offsprings is None or not only_offsprings or offsprings > popsize
        if only_offsprings:
            assert offsprings is not None, "only_offsprings only work if offsprings is not None (non-DE mode)"
        assert 0 <= recombination_ratio <= 1
        self.recombination_ratio = recombination_ratio
        self.popsize = popsize
        self.offsprings = offsprings
        self.only_offsprings = only_offsprings
        # self.de_step = de_step
        super().__init__()


RecES = EvolutionStrategy(recombination_ratio=1, only_offsprings=True, offsprings=60).with_name("RecES", register=True)
RecMixES = EvolutionStrategy(recombination_ratio=1, only_offsprings=False, offsprings=20).with_name("RecMixES", register=True)
RecMutDE = EvolutionStrategy(recombination_ratio=1, only_offsprings=False, offsprings=None).with_name("RecMutDE", register=True)
ES = EvolutionStrategy(recombination_ratio=0, only_offsprings=True, offsprings=60).with_name("ES", register=True)
MixES = EvolutionStrategy(recombination_ratio=0, only_offsprings=False, offsprings=20).with_name("MixES", register=True)
MutDE = EvolutionStrategy(recombination_ratio=0, only_offsprings=False, offsprings=None).with_name("MutDE", register=True)
