# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Callable, Dict, List
import numpy as np
from bayes_opt import BayesianOptimization
from scipy import optimize as scipyoptimize
from scipy import stats
from . import base
from .base import registry
from . import recaster
from . import sequences


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


@registry.register
class BO(recaster.SequentialRecastOptimizer):
    def __init__(self, dimension, budget=None, num_workers=1):
        super(BO, self).__init__(dimension, budget=budget, num_workers=num_workers)
        self.qr = "none"

    def _dirty_optimization(self, objective_function, num_workers):
        return self._internal_optimize(objective_function, num_workers)

    def get_optimization_function(self) -> Callable:
        # create a different sub-instance, so that the current instance is not referenced by the thread
        # (consequence: do not create a thread at initialization, or we get a thread explosion)
        subinstance = self.__class__(dimension=self.dimension, budget=self.budget, num_workers=self.num_workers)
        return subinstance._optimization_function  # type: ignore

    def _optimization_function(self, objective_function: Callable[[base.ArrayLike], float]) -> base.ArrayLike:

        def my_obj(**kwargs):
            v = [stats.norm.ppf(kwargs[str(i)]) for i in range(self.dimension)]
            v = [min(max(v_, -100), 100) for v_ in v]
            return -objective_function(v)   # We minimize!
        bounds = {}
        for i in range(self.dimension):
            bounds[str(i)] = (0., 1.)
        bo = BayesianOptimization(my_obj, bounds)
        if self.qr != "none":
            points_dict: Dict[str, List[base.ArrayLike]] = {}
            for i in range(self.dimension):
                points_dict[str(i)] = []
            budget = int(np.sqrt(self.budget))
            sampler: Optional[sequences.Sampler] = None
            if self.qr == "qr":
                sampler = sequences.ScrHammersleySampler(self.dimension, budget=budget)
            elif self.qr == "mqr":
                sampler = sequences.ScrHammersleySampler(self.dimension, budget=budget - 1)
            elif self.qr == "lhs":
                sampler = sequences.LHSSampler(self.dimension, budget=budget)
            elif self.qr == "r":
                sampler = sequences.RandomSampler(self.dimension, budget=budget)
            assert sampler is not None
            for i in range(budget):
                if self.qr == "mqr" and not i:
                    s = [0.5] * self.dimension
                else:
                    s = list(sampler())
                assert len(s) == self.dimension
                for j in range(self.dimension):
                    points_dict[str(j)].append(s[j])
            bo.explore(points_dict)
        assert budget is not None
        assert self.budget is not None
        budget = self.budget - (budget if self.qr != "none" else 0)
        ip = 1 if self.qr == "none" else 0
        bo.maximize(n_iter=budget - ip, init_points=ip)
        # print [bo.res['max']['max_params'][str(i)] for i in xrange(self.dimension)]
        v = [stats.norm.ppf(bo.res['max']['max_params'][str(i)]) for i in range(self.dimension)]
        v = [min(max(v_, -100), 100) for v_ in v]
        return v


@registry.register
class RBO(BO):
    def __init__(self, dimension, budget=None, num_workers=1):
        super(RBO, self).__init__(dimension, budget=budget, num_workers=num_workers)
        self.qr = "r"


@registry.register
class QRBO(BO):
    def __init__(self, dimension, budget=None, num_workers=1):
        super(QRBO, self).__init__(dimension, budget=budget, num_workers=num_workers)
        self.qr = "qr"


@registry.register
class MidQRBO(BO):
    def __init__(self, dimension, budget=None, num_workers=1):
        super(MidQRBO, self).__init__(dimension, budget=budget, num_workers=num_workers)
        self.qr = "mqr"


@registry.register
class LBO(BO):
    def __init__(self, dimension, budget=None, num_workers=1):
        super(LBO, self).__init__(dimension, budget=budget, num_workers=num_workers)
        self.qr = "lhs"
