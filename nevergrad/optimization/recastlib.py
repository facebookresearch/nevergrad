# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import functools
import math
import warnings
import weakref
import numpy as np
from scipy import optimize as scipyoptimize
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from nevergrad.common import errors
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
        return functools.partial(self._optimization_function, weakref.proxy(self))

    @staticmethod
    def _optimization_function(
        weakself: tp.Any, objective_function: tp.Callable[[tp.ArrayLike], float]
    ) -> tp.ArrayLike:
        # pylint:disable=unused-argument
        budget = np.inf if weakself.budget is None else weakself.budget
        best_res = np.inf
        best_x: np.ndarray = weakself.current_bests["average"].x  # np.zeros(self.dimension)
        if weakself.initial_guess is not None:
            best_x = np.array(weakself.initial_guess, copy=True)  # copy, just to make sure it is not modified
        remaining: float = budget - weakself._num_ask
        while remaining > 0:  # try to restart if budget is not elapsed
            options: tp.Dict[str, tp.Any] = {} if weakself.budget is None else {"maxiter": remaining}
            res = scipyoptimize.minimize(
                objective_function,
                best_x if not weakself.random_restart else weakself._rng.normal(0.0, 1.0, weakself.dimension),
                method=weakself.method,
                options=options,
                tol=0,
            )
            if res.fun < best_res:
                best_res = res.fun
                best_x = res.x
            remaining = budget - weakself._num_ask
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
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        # configuration
        self.algorithm = algorithm
        self._no_hypervolume = True
        self._initial_seed = -1

    def get_optimization_function(self) -> tp.Callable[[tp.Callable[..., tp.Any]], tp.Optional[tp.ArrayLike]]:
        if self._initial_seed == -1:
            self._initial_seed = self._rng.randint(2 ** 30)
        return functools.partial(self._optimization_function, weakref.proxy(self))
        # pylint:disable=useless-return

    @staticmethod
    def _optimization_function(
        weakself: tp.Any, objective_function: tp.Callable[[tp.ArrayLike], float]
    ) -> tp.Optional[tp.ArrayLike]:
        # pylint:disable=unused-argument, import-outside-toplevel
        from pymoo import optimize as pymoooptimize

        from pymoo.factory import get_algorithm as get_pymoo_algorithm

        # from pymoo.factory import get_reference_directions

        # reference direction code for when we want to use the other MOO optimizers in Pymoo
        # if self.algorithm in [
        #     "rnsga2",
        #     "nsga3",
        #     "unsga3",
        #     "rnsga3",
        #     "moead",
        #     "ctaea",
        # ]:  # algorithms that require reference points or reference directions
        #     the appropriate n_partitions must be looked into
        #     ref_dirs = get_reference_directions("das-dennis", self.num_objectives, n_partitions=12)
        #     algorithm = get_pymoo_algorithm(self.algorithm, ref_dirs)
        # else:
        algorithm = get_pymoo_algorithm(weakself.algorithm)
        problem = _create_pymoo_problem(weakself, objective_function)
        pymoooptimize.minimize(problem, algorithm, seed=weakself._initial_seed)
        return None

    def _internal_ask_candidate(self) -> p.Parameter:
        """
        Special version to make sure that num_objectives has been set before
        the proper _internal_ask_candidate, in our parent class, is called.
        """
        if self.num_objectives == 0:
            # dummy ask i.e. not activating pymoo until num_objectives is set
            warnings.warn(
                "with this optimizer, it is more efficient to set num_objectives before the optimization begins",
                errors.NevergradRuntimeWarning,
            )
            # We need to get a datapoint that is a random point in parameter space,
            # and waste an evaluation on it.
            return self.parametrization.spawn_child()
        return super()._internal_ask_candidate()

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        """
        Special version to make sure that we the extra initial evaluation which
        we may have done in order to get num_objectives, is discarded.
        Note that this discarding means that the extra point will not make it into
        replay_archive_tell. Correspondingly, because num_objectives will make it into
        the pickle, __setstate__ will never need a dummy ask.
        """
        if self._messaging_thread is None:
            return  # dummy tell i.e. not activating pymoo until num_objectives is set
        super()._internal_tell_candidate(candidate, loss)

    def _post_loss(self, candidate: p.Parameter, loss: float) -> tp.Loss:
        # pylint: disable=unused-argument
        """
        Multi-Objective override for this function.
        """
        return candidate.losses


class Pymoo(base.ConfiguredOptimizer):
    """Wrapper over Pymoo optimizer implementations, in standard ask and tell format.
    This is actually an import from Pymoo Optimize.

    Parameters
    ----------
    algorithm: str

        Use "algorithm-name" with following names to access algorithm classes:
        Single-Objective
        -"de"
        -'ga'
        -"brkga"
        -"nelder-mead"
        -"pattern-search"
        -"cmaes"
        Multi-Objective
        -"nsga2"
        Multi-Objective requiring reference directions, points or lines
        -"rnsga2"
        -"nsga3"
        -"unsga3"
        -"rnsga3"
        -"moead"
        -"ctaea"

    Note
    ----
    These optimizers do not support asking several candidates in a row
    """

    recast = True
    no_parallelization = True

    # pylint: disable=unused-argument
    def __init__(self, *, algorithm: str) -> None:
        super().__init__(_PymooMinimizeBase, locals())


class _PymooBatchMinimizeBase(recaster.BatchRecastOptimizer):

    # pylint: disable=abstract-method

    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        *,
        algorithm: str,
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        # configuration
        self.algorithm = algorithm
        self._no_hypervolume = True
        self._initial_seed = -1

    def get_optimization_function(self) -> tp.Callable[[tp.Callable[..., tp.Any]], tp.Optional[tp.ArrayLike]]:
        if self._initial_seed == -1:
            self._initial_seed = self._rng.randint(2 ** 30)
        return functools.partial(self._optimization_function, weakref.proxy(self))
        # pylint:disable=useless-return

    @staticmethod
    def _optimization_function(
        weakself: tp.Any, objective_function: tp.Callable[[tp.ArrayLike], float]
    ) -> tp.Optional[tp.ArrayLike]:
        # pylint:disable=unused-argument, import-outside-toplevel
        from pymoo import optimize as pymoooptimize

        from pymoo.factory import get_algorithm as get_pymoo_algorithm

        # from pymoo.factory import get_reference_directions

        # reference direction code for when we want to use the other MOO optimizers in Pymoo
        # if self.algorithm in [
        #     "rnsga2",
        #     "nsga3",
        #     "unsga3",
        #     "rnsga3",
        #     "moead",
        #     "ctaea",
        # ]:  # algorithms that require reference points or reference directions
        #     the appropriate n_partitions must be looked into
        #     ref_dirs = get_reference_directions("das-dennis", self.num_objectives, n_partitions=12)
        #     algorithm = get_pymoo_algorithm(self.algorithm, ref_dirs)
        # else:
        algorithm = get_pymoo_algorithm(weakself.algorithm)
        problem = _create_pymoo_problem(weakself, objective_function, False)
        pymoooptimize.minimize(problem, algorithm, seed=weakself._initial_seed)
        return None

    def _internal_ask_candidate(self) -> p.Parameter:
        """Reads messages from the thread in which the underlying optimization function is running
        New messages are sent as "ask".
        """
        # get a datapoint that is a random point in parameter space
        if self.num_objectives == 0:  # dummy ask i.e. not activating pymoo until num_objectives is set
            warnings.warn(
                "with this optimizer, it is more efficient to set num_objectives before the optimization begins",
                errors.NevergradRuntimeWarning,
            )
            return self.parametrization.spawn_child()
        return super()._internal_ask_candidate()

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        """Returns value for a point which was "asked"
        (none asked point cannot be "tell")
        """
        if self._messaging_thread is None:
            return  # dummy tell i.e. not activating pymoo until num_objectives is set
        super()._internal_tell_candidate(candidate, loss)

    def _post_loss(self, candidate: p.Parameter, loss: float) -> tp.Loss:
        # pylint: disable=unused-argument
        """
        Multi-Objective override for this function.
        """
        return candidate.losses


class PymooBatch(base.ConfiguredOptimizer):
    """Wrapper over Pymoo optimizer implementations, in standard ask and tell format.
    This is actually an import from Pymoo Optimize.

    Parameters
    ----------
    algorithm: str

        Use "algorithm-name" with following names to access algorithm classes:
        Single-Objective
        -"de"
        -'ga'
        -"brkga"
        -"nelder-mead"
        -"pattern-search"
        -"cmaes"
        Multi-Objective
        -"nsga2"
        Multi-Objective requiring reference directions, points or lines
        -"rnsga2"
        -"nsga3"
        -"unsga3"
        -"rnsga3"
        -"moead"
        -"ctaea"

    Note
    ----
    These optimizers do not support asking several candidates in a row
    """

    recast = True

    # pylint: disable=unused-argument
    def __init__(self, *, algorithm: str) -> None:
        super().__init__(_PymooBatchMinimizeBase, locals())


def _create_pymoo_problem(
    optimizer: base.Optimizer,
    objective_function: tp.Callable[[tp.ArrayLike], float],
    elementwise: bool = True,
):
    kwargs = {}
    try:
        # pylint:disable=import-outside-toplevel
        from pymoo.core.problem import ElementwiseProblem, Problem  # type: ignore

        Base = ElementwiseProblem if elementwise else Problem
    except ImportError:
        # Used if pymoo < 0.5.0
        # pylint:disable=import-outside-toplevel
        from pymoo.model.problem import Problem as Base  # type: ignore

        kwargs = {"elementwise_evaluation": elementwise}

    class _PymooProblem(Base):  # type: ignore
        def __init__(self, optimizer, objective_function):
            self.objective_function = objective_function
            super().__init__(
                n_var=optimizer.dimension,
                n_obj=optimizer.num_objectives,
                n_constr=0,  # constraints handled already by nevergrad
                xl=-math.pi * 0.5,
                xu=math.pi * 0.5,
                **kwargs,
            )

        def _evaluate(self, X, out, *args, **kwargs):
            # pylint:disable=unused-argument
            # pymoo is supplying us with bounded parameters in [-pi/2,pi/2]. Nevergrad wants unbounded reals from us.
            out["F"] = self.objective_function(np.tan(X))

    return _PymooProblem(optimizer, objective_function)


PymooNSGA2 = Pymoo(algorithm="nsga2").set_name("PymooNSGA2", register=True)
PymooBatchNSGA2 = PymooBatch(algorithm="nsga2").set_name("PymooBatchNSGA2", register=False)
