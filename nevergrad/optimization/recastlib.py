# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# verify
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Callable, Dict, List, Any
import numpy as np
from bayes_opt import BayesianOptimization
from scipy import optimize as scipyoptimize
from scipy import stats
from . import base
from . import recaster
from . import sequences


class _ScipyMinimizeBase(recaster.SequentialRecastOptimizer):

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self._parameters = ScipyOptimizer()
        self.multirun = 1  # work in progress
        self.initial_guess: Optional[base.ArrayLike] = None

    def get_optimization_function(self) -> Callable[[Callable[[base.ArrayLike], float]], base.ArrayLike]:
        # create a different sub-instance, so that the current instance is not referenced by the thread
        # (consequence: do not create a thread at initialization, or we get a thread explosion)
        subinstance = self.__class__(dimension=self.dimension, budget=self.budget, num_workers=self.num_workers)
        subinstance._parameters = self._parameters
        return subinstance._optimization_function

    def _optimization_function(self, objective_function: Callable[[base.ArrayLike], float]) -> base.ArrayLike:
        # pylint:disable=unused-argument
        budget = np.inf if self.budget is None else self.budget
        best_res = np.inf
        best_x: np.ndarray = np.zeros(self.dimension)
        if self.initial_guess is not None:
            best_x = np.array(self.initial_guess, copy=True)  # copy, just to make sure it is not modified
        remaining = budget - self._num_ask
        while remaining > 0:  # try to restart if budget is not elapsed
            options: Dict[str, int] = {} if self.budget is None else {"maxiter": remaining}
            res = scipyoptimize.minimize(objective_function, best_x if not self._parameters.random_restart else
                                         np.random.normal(0., 1., self.dimension), method=self._parameters.method, options=options, tol=0)
            if res.fun < best_res:
                best_res = res.fun
                best_x = res.x
            remaining = budget - self._num_ask
        return best_x


class ScipyOptimizer(base.ParametrizedFamily):
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

    _optimizer_class = _ScipyMinimizeBase

    def __init__(self, *, method: str = "Nelder-Mead", random_restart: bool = False) -> None:
        assert method in ["Nelder-Mead", "COBYLA", "SLSQP", "Powell"], f"Unknown method '{method}'"
        self.method = method
        self.random_restart = random_restart
        super().__init__()


NelderMead = ScipyOptimizer(method="Nelder-Mead").with_name("NelderMead", register=True)
Powell = ScipyOptimizer(method="Powell").with_name("Powell", register=True)
RPowell = ScipyOptimizer(method="Powell", random_restart=True).with_name("RPowell", register=True)
Cobyla = ScipyOptimizer(method="COBYLA").with_name("Cobyla", register=True)
RCobyla = ScipyOptimizer(method="COBYLA", random_restart=True).with_name("RCobyla", register=True)
SQP = ScipyOptimizer(method="SLSQP").with_name("SQP", register=True)
RSQP = ScipyOptimizer(method="SLSQP", random_restart=True).with_name("RSQP", register=True)


class _BO(recaster.SequentialRecastOptimizer):

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(dimension, budget=budget, num_workers=num_workers)
        self._parameters = ParametrizedBO()

    def get_optimization_function(self) -> Callable[[Callable[[base.ArrayLike], float]], base.ArrayLike]:
        # create a different sub-instance, so that the current instance is not referenced by the thread
        # (consequence: do not create a thread at initialization, or we get a thread explosion)
        subinstance = self.__class__(dimension=self.dimension, budget=self.budget, num_workers=self.num_workers)
        subinstance._parameters = self._parameters
        return subinstance._optimization_function

    def _optimization_function(self, objective_function: Callable[[base.ArrayLike], float]) -> base.ArrayLike:

        def my_obj(**kwargs: Any) -> float:
            v = [stats.norm.ppf(kwargs[str(i)]) for i in range(self.dimension)]
            v = [min(max(v_, -100.), 100.) for v_ in v]
            return -objective_function(v)   # We minimize!

        bounds = {}
        for i in range(self.dimension):
            bounds[str(i)] = (0., 1.)
        bo = BayesianOptimization(my_obj, bounds)
        if self._parameters.qr != "none":
            points_dict: Dict[str, List[base.ArrayLike]] = {}
            for i in range(self.dimension):
                points_dict[str(i)] = []
            budget = int(np.sqrt(self.budget))
            sampler: Optional[sequences.Sampler] = None
            if self._parameters.qr == "qr":
                sampler = sequences.HammersleySampler(self.dimension, budget=budget, scrambling=True)
            elif self._parameters.qr == "mqr":
                sampler = sequences.HammersleySampler(self.dimension, budget=budget - 1, scrambling=True)
            elif self._parameters.qr == "lhs":
                sampler = sequences.LHSSampler(self.dimension, budget=budget)
            elif self._parameters.qr == "r":
                sampler = sequences.RandomSampler(self.dimension, budget=budget)
            assert sampler is not None
            for i in range(budget):
                if self._parameters.qr == "mqr" and not i:
                    s = [0.5] * self.dimension
                else:
                    s = list(sampler())
                assert len(s) == self.dimension
                for j in range(self.dimension):
                    points_dict[str(j)].append(s[j])  # type: ignore
            bo.explore(points_dict)
        assert self.budget is not None
        budget = self.budget - (budget if self._parameters.qr != "none" else 0)
        ip = 1 if self._parameters.qr == "none" else 0
        bo.maximize(n_iter=budget - ip, init_points=ip)
        # print [bo.res['max']['max_params'][str(i)] for i in xrange(self.dimension)]
        v = [stats.norm.ppf(bo.res['max']['max_params'][str(i)]) for i in range(self.dimension)]
        v = [min(max(v_, -100.), 100.) for v_ in v]
        return v


class ParametrizedBO(base.ParametrizedFamily):
    """Bayesian optimization

    qr: str
        TODO
    """

    recast = True
    _optimizer_class = _BO

    def __init__(self, *, qr: str = "none") -> None:
        assert qr in ["r", "qr", "mqr", "lhs", "none"]
        self.qr = qr
        super().__init__()


BO = ParametrizedBO().with_name("BO", register=True)
RBO = ParametrizedBO(qr="r").with_name("RBO", register=True)
QRBO = ParametrizedBO(qr="qr").with_name("QRBO", register=True)
MidQRBO = ParametrizedBO(qr="mqr").with_name("MidQRBO", register=True)
LBO = ParametrizedBO(qr="lhs").with_name("LBO", register=True)
