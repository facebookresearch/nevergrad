# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Callable
import numpy as np
from scipy import optimize as scipyoptimize
from . import base
from .base import registry
from . import recaster


class ScipyMinimizeBase(recaster.SequentialRecastOptimizer):

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1, method: Optional[str] = None) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self.method = method
        self.multirun = 1  # work in progress
        assert self.method is not None, "A method must be specified"

    def get_optimization_function(self) -> Callable:
        # create a different sub-instance, so that the current instance is not referenced by the thread
        # (consequence: do not create a thread at initialization, or we get a thread explosion)
        subinstance = self.__class__(dimension=self.dimension, budget=self.budget, num_workers=self.num_workers)
        return subinstance._optimization_function  # type: ignore

    def _optimization_function(self, objective_function: Callable[[base.ArrayLike], float]) -> base.ArrayLike:
        # pylint:disable=unused-argument
        budget = np.inf if self.budget is None else self.budget
        best_res = np.inf
        best_x = np.zeros(self.dimension)
        remaining = budget - self._num_suggestions
        while remaining > 0:  # try to restart if budget is not elapsed
            options: dict = {} if self.budget is None else {"maxiter": remaining}
            res = scipyoptimize.minimize(objective_function, best_x, method=self.method, options=options, tol=0)
            if res.fun < best_res:
                best_res = res.fun
                best_x = res.x
            remaining = budget - self._num_suggestions
        return best_x


@registry.register
class NelderMead(ScipyMinimizeBase):

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers, method="Nelder-Mead")


@registry.register
class Powell(ScipyMinimizeBase):
    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget, num_workers=num_workers, method="Powell")


@registry.register
class Cobyla(ScipyMinimizeBase):
    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget, num_workers=num_workers, method="COBYLA")


@registry.register
class SQP(ScipyMinimizeBase):
    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget, num_workers=num_workers, method="SLSQP")
