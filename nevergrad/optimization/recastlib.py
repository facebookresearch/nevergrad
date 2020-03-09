# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# verify
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Callable, Dict
import numpy as np
from scipy import optimize as scipyoptimize
from nevergrad.parametrization import parameter as p
from . import base
from .base import IntOrParameter
from . import recaster


class _ScipyMinimizeBase(recaster.SequentialRecastOptimizer):

    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: Optional[int] = None,
        num_workers: int = 1,
        *,
        method: str = "Nelder-Mead",
        random_restart: bool = False
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.multirun = 1  # work in progress
        self.initial_guess: Optional[base.ArrayLike] = None
        # configuration
        assert method in ["Nelder-Mead", "COBYLA", "SLSQP", "Powell"], f"Unknown method '{method}'"
        self.method = method
        self.random_restart = random_restart

#    def _internal_tell_not_asked(self, x: base.ArrayLike, value: float) -> None:
    def _internal_tell_not_asked(self, candidate: p.Parameter, value: float) -> None:
        """Called whenever calling "tell" on a candidate that was not "asked".
        Defaults to the standard tell pipeline.
        """  # We do not do anything; this just updates the current best.

    def get_optimization_function(self) -> Callable[[Callable[[base.ArrayLike], float]], base.ArrayLike]:
        # create a different sub-instance, so that the current instance is not referenced by the thread
        # (consequence: do not create a thread at initialization, or we get a thread explosion)
        subinstance = self.__class__(
            parametrization=self.parametrization,
            budget=self.budget,
            num_workers=self.num_workers,
            method=self.method,
            random_restart=self.random_restart)
        return subinstance._optimization_function

    def _optimization_function(self, objective_function: Callable[[base.ArrayLike], float]) -> base.ArrayLike:
        # pylint:disable=unused-argument
        budget = np.inf if self.budget is None else self.budget
        best_res = np.inf
        best_x: np.ndarray = self.current_bests["average"].x  # np.zeros(self.dimension)
        if self.initial_guess is not None:
            best_x = np.array(self.initial_guess, copy=True)  # copy, just to make sure it is not modified
        remaining = budget - self._num_ask
        while remaining > 0:  # try to restart if budget is not elapsed
            options: Dict[str, int] = {} if self.budget is None else {"maxiter": remaining}
            res = scipyoptimize.minimize(
                objective_function,
                best_x if not self.random_restart else self._rng.normal(0.0, 1.0, self.dimension),
                method=self.method,
                options=options,
                tol=0,
            )
            if res.fun < best_res:
                best_res = res.fun
                best_x = res.x
            remaining = budget - self._num_ask
        return best_x


class ScipyOptimizer(base.ConfiguredOptimizer):
    """Scripy optimizers in a ask and tell format

    Parameters
    ----------
    method: str
        Name of the method to use, among Nelder-Mead, COBYLA, SLSQP and Powell
    random_restart: bool
        whether to restart at a random point if the optimizer converged but the budget is not entirely
        spent yet (otherwise, restarts from best point)
    """

    recast = True
    no_parallelization = True

    # pylint: disable=unused-argument
    def __init__(
        self,
        *,
        method: str = "Nelder-Mead",
        random_restart: bool = False
    ) -> None:
        super().__init__(_ScipyMinimizeBase, locals())


NelderMead = ScipyOptimizer(method="Nelder-Mead").set_name("NelderMead", register=True)
Powell = ScipyOptimizer(method="Powell").set_name("Powell", register=True)
RPowell = ScipyOptimizer(method="Powell", random_restart=True).set_name("RPowell", register=True)
Cobyla = ScipyOptimizer(method="COBYLA").set_name("Cobyla", register=True)
RCobyla = ScipyOptimizer(method="COBYLA", random_restart=True).set_name("RCobyla", register=True)
SQP = ScipyOptimizer(method="SLSQP").set_name("SQP", register=True)
SLSQP = SQP  # Just so that people who are familiar with SLSQP naming are not lost.
RSQP = ScipyOptimizer(method="SLSQP", random_restart=True).set_name("RSQP", register=True)
RSLSQP = RSQP  # Just so that people who are familiar with SLSQP naming are not lost.
