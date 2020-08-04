
from functools import partialmethod
from itertools import cycle
import typing as tp

from deap import tools
import numpy as np

from . import base
from ..parametrization import parameter as p
from ..common.typetools import ArrayLike


class Mutation:
    def __init__(
        self,
        random_state: np.random.RandomState,
        mutation: str,
        mutation_args: tp.Optional[tp.Mapping]
    ):
        pass


class Crossover:
    def __init__(
        self,
        random_state: np.random.RandomState,
        crossover: str,
        crossover_args: tp.Optional[tp.Mapping] = None
    ):
        pass


# def _resursive_find_groups(parametrization: p.Parameter):
#     for param in parametrization.value:
#         if isinstance(param, (p.Tuple)):


#     current_groups =

def split_candidate(groups: tp.List[int], values: ArrayLike):
    pass


class _CooperativeOptimization(base.Optimizer):
    def __init__(
        self,
        parametrization: base.IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        *,
        config: tp.Optional["CooperativeOptimization"] = None
    ):
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self._config = CooperativeOptimization() if config is None else config
        self.num_species = 10
        pop_choice = {"standard": 0, "dimension": self.dimension + 1, "large": 7 * self.dimension}
        if isinstance(self._config.popsize, int):
            self.llambda = self._config.popsize
        else:
            self.llambda = max(30, self.num_workers, pop_choice[self._config.popsize])

        # TODO: Initialize groups strategically
        if self._config.groups is None:
            self.species = list(self.parametrization._content.values())

        self.species_iterator = cycle(range(len(self.species)))
        self._uid_queue = base.utils.UidQueue()
        self.populations: tp.List[tp.List[p.Parameter]] = [
            [] for _ in range(len(self.species))
        ]
        self.candidates : tp.Dict[str, p.Parameter]
        self.representatives: tp.List[p.Parameter] = []

    def _internal_ask_candidate(self) -> p.Parameter:
        idx = next(self.species_iterator)
        current_species = self.species[idx]
        current_population = self.populations[idx]

        # Generate a representative of each species
        if not self.representatives:
            for sp, pop in zip(self.species, self.populations):
                init_values = self._rng.normal(0, 1, sp.dimension)
                rep = sp.spawn_child().set_standardized_data(init_values)
                rep.heritage["lineage"] = rep.uid  # new lineage
                self.representatives.append(rep)
                pop[rep.uid] = rep

            # TODO: Make this line work for Instrumentation
            candidate_value = tuple(v.value for v in self.representatives)

        # Random initialization
        elif len(current_population) < self.llambda:  # initialization phase
            init_values = self._rng.normal(0, 1, current_species.dimension)
            candidate = current_species.spawn_child().set_standardized_data(init_values)

            candidate.heritage["lineage"] = candidate.uid  # new lineage
            current_population.append(candidate)
            self.candidates[candidate.uid] = candidate
            self._uid_queue.asked.add(candidate.uid)

            # TODO: Make this work with Instrumentation
            candidate_value = tuple(v.value for v in self.representatives[:idx]) \
                              + (candidate.value,) \
                              + tuple(v.value for v in self.representatives[idx+1:])

        # Crossover or mutate
        else:
            pass


        return self.parametrization.spawn_child(new_value=candidate_value)

    def _internal_tell_candidate(self, candidate: p.Parameter, value: float) -> None:
        pass

    def _internal_tell_not_asked(self, candidate: p.Parameter, value: float) -> None:
        pass


class CooperativeOptimization(base.ConfiguredOptimizer, ):
    def __init__(
        self,
        *,
        groups: tp.Union[str, tp.List[tp.Tuple[tp.Union[int, str]]], None] = None,
        crossover: str = "sbx",
        mutation: str = "polynomial",
        popsize: tp.Union[str, int] = "standard"
    ):
        super().__init__(_CooperativeOptimization, locals(), as_config=True)
        self.groups = groups
        self.crossover = crossover
        self.mutation = mutation
        self.popsize = popsize

Coop = CooperativeOptimization().set_name("Coop")