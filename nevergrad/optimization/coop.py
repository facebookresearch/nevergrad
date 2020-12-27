
from functools import partialmethod
from itertools import cycle
from operator import itemgetter

from deap import tools  # type: ignore
import numpy as np

from . import base
from .base import registry
from . import optimizerlib
from ..parametrization import parameter as p
from ..common import typing as tp
from ..parametrization import helpers as paramhelpers


OptimizerOrListOfOptimizer = tp.Union[base.OptCls, tp.Sequence[base.OptCls]]

@registry.register
class CoopOptimizer(optimizerlib.SplitOptimizer):
    def __init__(
            self,
            parametrization: base.IntOrParameter,
            budget: tp.Optional[int] = None,
            num_workers: int = 1,
            num_optims: tp.Optional[int] = None,
            num_vars: tp.Optional[tp.List[int]] = None,
            multivariate_optimizer: base.OptCls = optimizerlib.CMA,
            monovariate_optimizer: base.OptCls = optimizerlib.OnePlusOne,
            non_deterministic_descriptor: bool = True,
            optimizer_selection: str = "None"
    ) -> None:
        super().__init__(
            parametrization=parametrization,
            budget=budget,
            num_workers=num_workers,
            num_optims=num_optims,
            num_vars=num_vars,
            multivariate_optimizer=multivariate_optimizer,
            monovariate_optimizer=monovariate_optimizer,
            progressive=False,
            non_deterministic_descriptor=non_deterministic_descriptor
        )

        # TODO: Add bandit selection
        self.next_optimizer_index = cycle(range(len(self.optims)))

    def _internal_ask_candidate(self) -> p.Parameter:
        candidates: tp.List[tp.Tuple[p.Parameter, bool]] = []
        current_optim_index = next(self.next_optimizer_index)
        initialization = self.num_ask < len(self.optims)

        for i, opt in enumerate(self.optims):
            if i == current_optim_index or initialization:
                # Ask new candidate from current optimizer
                # On True, accept new fitness
                candidates.append((opt.ask(), True))
            else:
                # Take best candidates from other optimizers unless
                # initialization phase
                # On False, keep old fitness
                candidates.append((opt.recommend(), False))


        data = np.concatenate([c.get_standardized_data(reference=opt.parametrization)
                               for (c, _), opt in zip(candidates, self.optims)], axis=0)
        cand = self.parametrization.spawn_child().set_standardized_data(data)

        # Filter candidates that should not be evaluated
        self._subcandidates[cand.uid] = [c if b else None for c, b in candidates]
        return cand

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        candidates = self._subcandidates.pop(candidate.uid)
        for cand, opt in zip(candidates, self.optims):
            # Make sure we tell the right candidates
            if cand is not None:
                opt.tell(cand, loss)


# class CooperativeOptimization(base.ConfiguredOptimizer, ):
#     def __init__(
#         self,
#         *,
#         groups: tp.Union[str, tp.List[tp.Tuple[tp.Union[int, str]]], None] = None,
#         crossover: str = "sbx",
#         mutation: str = "polynomial",
#         popsize: tp.Union[str, int] = "standard"
#     ):
#         super().__init__(_CooperativeOptimization, locals(), as_config=True)
#         self.groups = groups
#         self.crossover = crossover
#         self.mutation = mutation
#         self.popsize = popsize


# Coop = CoopOptimizer().set_name("Coop", register=True)
