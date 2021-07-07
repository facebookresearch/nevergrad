# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from scipy import optimize as scipyoptimize
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from . import base
from .base import IntOrParameter
from . import recaster

import threading

    
def smac_obj(objective_function, p):
    print(f"SMAC proposes {p}")
    data = np.asarray([np.tan(np.pi * p[i] / 2.0) for i in range(len(p))], dtype=np.float)
    print(f"converted to {data}")
    res = 1. #objective_function(data)
    print(f"SMAC will receive {res}")
    return res


class _ScipyMinimizeBase(recaster.SequentialRecastOptimizer):
    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        *,
        method: str = "Nelder-Mead",
        random_restart: bool = False,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.multirun = 1  # work in progress
        self.initial_guess: tp.Optional[tp.ArrayLike] = None
        # configuration
        assert method in ["SMAC", "SMAC2", "Nelder-Mead", "COBYLA", "SLSQP", "Powell"], f"Unknown method '{method}'"
        self.method = method
        self.random_restart = random_restart

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.Loss) -> None:
        """Called whenever calling "tell" on a candidate that was not "asked".
        Defaults to the standard tell pipeline.
        """  # We do not do anything; this just updates the current best.

    def get_optimization_function(self) -> tp.Callable[[tp.Callable[[tp.ArrayLike], float]], tp.ArrayLike]:
        # create a different sub-instance, so that the current instance is not referenced by the thread
        # (consequence: do not create a thread at initialization, or we get a thread explosion)
        subinstance = self.__class__(
            parametrization=self.parametrization,
            budget=self.budget,
            num_workers=self.num_workers,
            method=self.method,
            random_restart=self.random_restart,
        )
        subinstance.archive = self.archive
        subinstance.current_bests = self.current_bests
        return subinstance._optimization_function

    def _optimization_function(self, objective_function: tp.Callable[[tp.ArrayLike], float]) -> tp.ArrayLike:
        # pylint:disable=unused-argument
        budget = np.inf if self.budget is None else self.budget
        best_res = np.inf
        best_x: np.ndarray = self.current_bests["average"].x
        if self.initial_guess is not None:
            best_x = np.array(self.initial_guess, copy=True)  # copy, just to make sure it is not modified
        remaining: float = budget - self._num_ask
        while remaining > 0:  # try to restart if budget is not elapsed
            options: tp.Dict[str, tp.Any] = {} if self.budget is None else {"maxiter": remaining}
            if self.method == "SMAC2":
                from ConfigSpace.hyperparameters import (
                    UniformFloatHyperparameter,
                )  # noqa  # pylint: disable=unused-import

                # Import ConfigSpace and different types of parameters
                from smac.configspace import ConfigurationSpace  # noqa  # pylint: disable=unused-import
                from smac.facade.smac_hpo_facade import SMAC4HPO  # noqa  # pylint: disable=unused-import

                # Import SMAC-utilities
                from smac.scenario.scenario import Scenario  # noqa  # pylint: disable=unused-import

                print(f"start SMAC2 optimization with budget {budget} in dimension {self.dimension}")
                cs = ConfigurationSpace()
                cs.add_hyperparameters(
                    [
                        UniformFloatHyperparameter(f"x{i}", -1.0, 1.0, default_value=0.0)
                        for i in range(self.dimension)
                    ]
                )
                scenario = Scenario(
                    {
                        "run_obj": "quality",  # we optimize quality (alternatively runtime)
                        "runcount-limit": budget,  # max. number of function evaluations
                        "cs": cs,  # configuration space
                        "deterministic": "true",
                    }
                )
                def smac2_obj(p):
                    print(f"SMAC2 proposes {p}")
                    p = np.asarray([p[f"x{i}"] for i in range(len(p.keys()))])
                    data = np.asarray(np.tan(np.pi * p / 2.0), dtype=np.float)
                    print(f"converted to {data}")
                    res = objective_function(data)  # Stuck here!
                    print(f"SMAC2 will receive {res}")
                    return res
                smac = SMAC4HPO(scenario=scenario, rng=self._rng.randint(5000), tae_runner=smac2_obj)
                res = smac.optimize()
                best_x = np.asarray(
                    [np.tan(np.pi * 0.5 * res[f"x{k}"]) for k in range(len(res.keys()))], dtype=np.float
                )

            elif self.method == "SMAC":
                import smac  # noqa  # pylint: disable=unused-import
                import scipy.optimize  # noqa  # pylint: disable=unused-import
                from smac.facade.func_facade import fmin_smac  # noqa  # pylint: disable=unused-import

                print(f"start SMAC optimization with budget {budget} in dimension {self.dimension}")
                assert budget is not None
                from functools import partial
                import time

                def thread_function(x):
                    for k in range(500):
                        print("pouet=", objective_function([1,2]))
                        time.sleep(1)

                thread = threading.Thread(target=thread_function, args=(1,))
                thread.start()

                x, cost, _ = fmin_smac(
                    func=partial(smac_obj, objective_function),
                    x0=[0.0] * self.dimension,
                    bounds=[(-1, 1)] * self.dimension,
                    maxfun=budget,
                    rng=self._rng.randint(5000),
                )
                thread.join()
                print("end SMAC optimization")

                if cost < best_res:
                    best_res = cost
                    best_x = np.tan(np.pi * x / 2.0)
            else:
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
    """Wrapper over Scipy optimizer implementations, in standard ask and tell format.
    This is actually an import from scipy-optimize, including Sequential Quadratic Programming,

    Parameters
    ----------
    method: str
        Name of the method to use among:

        - Nelder-Mead
        - COBYLA
        - SQP (or SLSQP): very powerful e.g. in continuous noisy optimization. It is based on
          approximating the objective function by quadratic models.
        - Powell
    random_restart: bool
        whether to restart at a random point if the optimizer converged but the budget is not entirely
        spent yet (otherwise, restarts from best point)

    Note
    ----
    These optimizers do not support asking several candidates in a row
    """

    recast = True
    no_parallelization = True

    # pylint: disable=unused-argument
    def __init__(self, *, method: str = "Nelder-Mead", random_restart: bool = False) -> None:
        super().__init__(_ScipyMinimizeBase, locals())


NelderMead = ScipyOptimizer(method="Nelder-Mead").set_name("NelderMead", register=True)
SMAC = ScipyOptimizer(method="SMAC").set_name("SMAC", register=True)
SMAC2 = ScipyOptimizer(method="SMAC2").set_name("SMAC2", register=True)
Powell = ScipyOptimizer(method="Powell").set_name("Powell", register=True)
RPowell = ScipyOptimizer(method="Powell", random_restart=True).set_name("RPowell", register=True)
Cobyla = ScipyOptimizer(method="COBYLA").set_name("Cobyla", register=True)
RCobyla = ScipyOptimizer(method="COBYLA", random_restart=True).set_name("RCobyla", register=True)
SQP = ScipyOptimizer(method="SLSQP").set_name("SQP", register=True)
SLSQP = SQP  # Just so that people who are familiar with SLSQP naming are not lost.
RSQP = ScipyOptimizer(method="SLSQP", random_restart=True).set_name("RSQP", register=True)
RSLSQP = RSQP  # Just so that people who are familiar with SLSQP naming are not lost.
