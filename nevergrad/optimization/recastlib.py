# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math
import numpy as np
from scipy import optimize as scipyoptimize
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from . import base
from .base import IntOrParameter
from . import recaster


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
        assert method in ["Nelder-Mead", "COBYLA", "SLSQP", "Powell"], f"Unknown method '{method}'"
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
        best_x: np.ndarray = self.current_bests["average"].x  # np.zeros(self.dimension)
        if self.initial_guess is not None:
            best_x = np.array(self.initial_guess, copy=True)  # copy, just to make sure it is not modified
        remaining: float = budget - self._num_ask
        while remaining > 0:  # try to restart if budget is not elapsed
            options: tp.Dict[str, tp.Any] = {} if self.budget is None else {"maxiter": remaining}
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
Powell = ScipyOptimizer(method="Powell").set_name("Powell", register=True)
RPowell = ScipyOptimizer(method="Powell", random_restart=True).set_name("RPowell", register=True)
Cobyla = ScipyOptimizer(method="COBYLA").set_name("Cobyla", register=True)
RCobyla = ScipyOptimizer(method="COBYLA", random_restart=True).set_name("RCobyla", register=True)
SQP = ScipyOptimizer(method="SLSQP").set_name("SQP", register=True)
SLSQP = SQP  # Just so that people who are familiar with SLSQP naming are not lost.
RSQP = ScipyOptimizer(method="SLSQP", random_restart=True).set_name("RSQP", register=True)
RSLSQP = RSQP  # Just so that people who are familiar with SLSQP naming are not lost.


class _PymooMinimizeBase(recaster.SequentialRecastOptimizer):
    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        *,
        algorithm: str,
        random_restart: bool = False,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.multirun = 1  # work in progress
        self.initial_guess: tp.Optional[tp.ArrayLike] = None
        # configuration
        self.algorithm = algorithm
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
            algorithm=self.algorithm,
            random_restart=self.random_restart,
        )
        if self.num_objectives > 0:
            subinstance.num_objectives = self.num_objectives
        else:
            raise RuntimeError(
                "This optimizer requires optimizer.num_objectives to be explicity set before the first call to ask."
            )
        subinstance.archive = self.archive
        subinstance.current_bests = self.current_bests
        return subinstance._optimization_function

    def _optimization_function(self, objective_function: tp.Callable[[tp.ArrayLike], float]) -> tp.ArrayLike:
        # pylint:disable=unused-argument
        # pylint:disable=import-outside-toplevel
        from pymoo import optimize as pymoooptimize  # type: ignore

        # pylint:disable=import-outside-toplevel
        from pymoo.factory import get_algorithm as get_pymoo_algorithm  # type: ignore

        # pylint:disable=import-outside-toplevel
        from pymoo.factory import get_reference_directions  # type: ignore

        budget = np.inf if self.budget is None else self.budget
        best_res = np.inf
        best_x: np.ndarray = self.current_bests["average"].x  # np.zeros(self.dimension)
        problem = self._create_pymoo_problem(self, objective_function)
        if self.algorithm in ["rnsga2", "nsga3", "unsga3", "rnsga3", "moead", "ctaea"]:
            ref_dirs = get_reference_directions("das-dennis", self.num_objectives, n_partitions=12)
            algorithm = get_pymoo_algorithm(self.algorithm, ref_dirs)
        else:
            algorithm = get_pymoo_algorithm(self.algorithm)
        if self.initial_guess is not None:
            best_x = np.array(self.initial_guess, copy=True)  # copy, just to make sure it is not modified
        remaining: float = budget - self._num_ask
        while remaining > 0:  # try to restart if budget is not elapsed
            res = pymoooptimize.minimize(problem, algorithm)
            if res.F < best_res:
                best_res = res.F
                best_x = res.X
            remaining = budget - self._num_ask
        return best_x

    def _internal_ask_candidate(self) -> p.Parameter:
        """Reads messages from the thread in which the underlying optimization function is running
        New messages are sent as "ask".
        """
        if self.num_objectives == 0:
            data = self._rng.normal(0, 1, self.dimension)
            candidate = self.parametrization.spawn_child().set_standardized_data(data)
            return candidate
        return super()._internal_ask_candidate()

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        """Returns value for a point which was "asked"
        (none asked point cannot be "tell")
        """
        x = candidate.get_standardized_data(reference=self.parametrization)
        if self._messaging_thread is None:
            return
        if not self._messaging_thread.is_alive():  # optimizer is done
            self._check_error()
            return
        messages = [m for m in self._messaging_thread.messages if m.meta.get("asked", False) and not m.done]
        messages = [m for m in messages if m.meta["uid"] == candidate.uid]
        if not messages:
            raise RuntimeError(f"No message for evaluated point {x}: {self._messaging_thread.messages}")
        # print(candidate.losses)
        messages[0].result = candidate.losses  # post all the losses, and the thread will deal with it

    def _create_pymoo_problem(
        self, optimizer: base.Optimizer, objective_function: tp.Callable[[tp.ArrayLike], float]
    ):
        # pylint:disable=import-outside-toplevel
        from pymoo.model.problem import Problem  # type: ignore

        class _PymooProblem(Problem):
            def __init__(self, optimizer, objective_function):
                self.objective_function = objective_function
                super().__init__(
                    n_var=optimizer.dimension,
                    n_obj=optimizer.num_objectives,
                    n_constr=0,
                    xl=-math.pi * 0.5,
                    xu=math.pi * 0.5,
                    elementwise_evaluation=True,
                )
                # print("num objectives", optimizer.num_objectives)

            def _evaluate(self, X, out, *args, **kwargs):
                # pylint:disable=unused-argument
                out["F"] = self.objective_function(np.tan(X))

        return _PymooProblem(optimizer, objective_function)


class PymooOptimizer(base.ConfiguredOptimizer):
    """Wrapper over Pymoo optimizer implementations, in standard ask and tell format.
    This is actually an import from Pymoo Optimize.

    Parameters
    ----------
    algorithm: PymooAlgorithm

        Use get_pymoo_algorithm("algorithm-name") with following names to access algorithm classes:
        -"de"
        -'ga'
        -"brkga"
        -"nelder-mead"
        -"pattern-search"
        -"cmaes"
        -"nsga2"
        -"rnsga2"
        -"nsga3"
        -"unsga3"
        -"rnsga3"
        -"moead"
        -"ctaea"

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
    def __init__(self, *, algorithm: str, random_restart: bool = False) -> None:
        super().__init__(_PymooMinimizeBase, locals())


# PymooNM = PymooOptimizer(algorithm=get_pymoo_algorithm("nelder-mead")).set_name("nelder-mead", register=True)
# Nelder-Mead is for single objective
# PymooDE = PymooOptimizer(algorithm=get_pymoo_algorithm("de")).set_name("de", register=True)
# PymooGA = PymooOptimizer(algorithm=get_pymoo_algorithm("ga")).set_name("ga", register=True)
# PymooBRKGA = PymooOptimizer(algorithm=get_pymoo_algorithm("brkga")).set_name("brkga", register=True)
# PymooCMAES = PymooOptimizer(algorithm=get_pymoo_algorithm("cmaes")).set_name("cmaes", register=True)
PymooNSGA2 = PymooOptimizer(algorithm="nsga2").set_name("PymooNSGA2", register=True)
# PymooRNSGA2 = PymooOptimizer(algorithm="rnsga2").set_name("PymooRNSGA2", register=True)
# PymooNSGA3 = PymooOptimizer(algorithm="nsga3").set_name("PymooNSGA3", register=True)
# PymooUNSGA3 = PymooOptimizer(algorithm="unsga3").set_name("PymooUNSGA3", register=True)
# PymooRNSGA3 = PymooOptimizer(algorithm="rnsga3").set_name("PymooRNSGA3", register=True)
# PymooMOEAD = PymooOptimizer(algorithm="moead").set_name("PymooMOEAD", register=True)
# PymooCTAEA = PymooOptimizer(algorithm="ctaea").set_name("PymooCTAEA", register=True)
