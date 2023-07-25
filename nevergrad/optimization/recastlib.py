# Copyright (c) Meta Platforms, Inc. and affiliates.
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


class _NonObjectMinimizeBase(recaster.SequentialRecastOptimizer):
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
        self._normalizer: tp.Any = None
        self.initial_guess: tp.Optional[tp.ArrayLike] = None
        # configuration
        assert (
            method
            in [
                "CmaFmin2",
                "Nelder-Mead",
                "COBYLA",
                "SLSQP",
                "Powell",
            ]
            or "NLOPT" in method
            or "BFGS" in method
        ), f"Unknown method '{method}'"
        self.method = method
        self.random_restart = random_restart
        # The following line rescales to [0, 1] if fully bounded.

        if method == "CmaFmin2" or "NLOPT" in method:
            normalizer = p.helpers.Normalizer(self.parametrization)
            self._normalizer = normalizer

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
            # options: tp.Dict[str, tp.Any] = {} if self.budget is None else {"maxiter": remaining}
            if weakself.method[:5] == "NLOPT":
                # This is NLOPT, used as in the PCSE simulator notebook.
                # ( https://github.com/ajwdewit/pcse_notebooks ).
                import nlopt

                def nlopt_objective_function(*args):
                    try:
                        data = np.asarray([arg for arg in args if len(arg) > 0])[0]
                    except Exception as e:
                        raise ValueError(f"{e}:\n{args}\n {[arg for arg in args]}")
                    assert len(data) == weakself.dimension, (
                        str(data) + " does not have length " + str(weakself.dimension)
                    )
                    if weakself._normalizer is not None:
                        data = weakself._normalizer.backward(np.asarray(data, dtype=np.float32))
                    return objective_function(data)

                # Sbplx (based on Subplex) is used by default.
                nlopt_param = (
                    getattr(nlopt, weakself.method[6:]) if len(weakself.method) > 5 else nlopt.LN_SBPLX
                )
                opt = nlopt.opt(nlopt_param, weakself.dimension)
                # Assign the objective function calculator
                opt.set_min_objective(nlopt_objective_function)
                # Set the bounds.
                opt.set_lower_bounds(np.zeros(weakself.dimension))
                opt.set_upper_bounds(np.ones(weakself.dimension))
                # opt.set_initial_step([0.05, 0.05])
                opt.set_maxeval(budget)

                # Start the optimization with the first guess
                firstguess = 0.5 * np.ones(weakself.dimension)
                best_x = opt.optimize(firstguess)
                # print("\noptimum at TDWI: %s, SPAN: %s" % (x[0], x[1]))
                # print("minimum value = ",  opt.last_optimum_value())
                # print("result code = ", opt.last_optimize_result())
                # print("With %i function calls" % objfunc_calculator.n_calls)
                if weakself._normalizer is not None:
                    best_x = weakself._normalizer.backward(np.asarray(best_x, dtype=np.float32))

            elif weakself.method == "CmaFmin2":
                import cma  # import inline in order to avoid matplotlib initialization warning

                def cma_objective_function(data):
                    # Hopefully the line below does nothing if unbounded and rescales from [0, 1] if bounded.
                    if weakself._normalizer is not None and weakself._normalizer.fully_bounded:
                        data = weakself._normalizer.backward(np.asarray(data, dtype=np.float32))
                    return objective_function(data)

                # cma.fmin2(objective_function, [0.0] * self.dimension, [1.0] * self.dimension, remaining)
                x0 = (
                    0.5 * np.ones(weakself.dimension)
                    if weakself._normalizer is not None and weakself._normalizer.fully_bounded
                    else np.zeros(weakself.dimension)
                )
                num_calls = 0
                while budget - num_calls > 0:
                    options = {"maxfevals": budget - num_calls, "verbose": -9}
                    if weakself._normalizer is not None and weakself._normalizer.fully_bounded:
                        # Tell CMA to work in [0, 1].
                        options["bounds"] = [0.0, 1.0]
                    res = cma.fmin(
                        cma_objective_function,
                        x0=x0,
                        sigma0=0.2,
                        options=options,
                        restarts=9,
                    )
                    x0 = (
                        0.5
                        + np.random.uniform() * np.random.uniform(low=-0.5, high=0.5, size=weakself.dimension)
                        if weakself._normalizer is not None and weakself._normalizer.fully_bounded
                        else np.random.randn(weakself.dimension)
                    )
                    if res[1] < best_res:
                        best_res = res[1]
                        best_x = res[0]
                        if weakself._normalizer is not None:
                            best_x = weakself._normalizer.backward(np.asarray(best_x, dtype=np.float32))
                    num_calls += res[2]
            else:
                res = scipyoptimize.minimize(
                    objective_function,
                    best_x
                    if not weakself.random_restart
                    else weakself._rng.normal(0.0, 1.0, weakself.dimension),
                    method=weakself.method,
                    options=options,
                    tol=0,
                )
                if res.fun < best_res:
                    best_res = res.fun
                    best_x = res.x
            remaining = budget - weakself._num_ask
        assert best_x is not None
        return best_x


class NonObjectOptimizer(base.ConfiguredOptimizer):
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
        - NLOPT* (https://nlopt.readthedocs.io/en/latest/; by default, uses Sbplx, based on Subplex);
            can be NLOPT,
                NLOPT_LN_SBPLX,
                NLOPT_LN_PRAXIS,
                NLOPT_GN_DIRECT,
                NLOPT_GN_DIRECT_L,
                NLOPT_GN_CRS2_LM,
                NLOPT_GN_AGS,
                NLOPT_GN_ISRES,
                NLOPT_GN_ESCH,
                NLOPT_LN_COBYLA,
                NLOPT_LN_BOBYQA,
                NLOPT_LN_NEWUOA_BOUND,
                NLOPT_LN_NELDERMEAD.
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
        super().__init__(_NonObjectMinimizeBase, locals())


NelderMead = NonObjectOptimizer(method="Nelder-Mead").set_name("NelderMead", register=True)
CmaFmin2 = NonObjectOptimizer(method="CmaFmin2").set_name("CmaFmin2", register=True)
NLOPT = NonObjectOptimizer(method="NLOPT").set_name("NLOPT", register=True)
Powell = NonObjectOptimizer(method="Powell").set_name("Powell", register=True)
RPowell = NonObjectOptimizer(method="Powell", random_restart=True).set_name("RPowell", register=True)
BFGS = NonObjectOptimizer(method="BFGS", random_restart=True).set_name("BFGS", register=True)
LBFGSB = NonObjectOptimizer(method="L-BFGS-B", random_restart=True).set_name("LBFGSB", register=True)
Cobyla = NonObjectOptimizer(method="COBYLA").set_name("Cobyla", register=True)
RCobyla = NonObjectOptimizer(method="COBYLA", random_restart=True).set_name("RCobyla", register=True)
SQP = NonObjectOptimizer(method="SLSQP").set_name("SQP", register=True)
SLSQP = SQP  # Just so that people who are familiar with SLSQP naming are not lost.
RSQP = NonObjectOptimizer(method="SLSQP", random_restart=True).set_name("RSQP", register=True)
RSLSQP = RSQP  # Just so that people who are familiar with SLSQP naming are not lost.
# NEWUOA = NonObjectOptimizer(method="NLOPT_LN_NEWUOA_BOUND").set_name("NEWUOA", register=True)
NLOPT_LN_SBPLX = NonObjectOptimizer(method="NLOPT_LN_SBPLX").set_name("NLOPT_LN_SBPLX", register=True)
NLOPT_LN_PRAXIS = NonObjectOptimizer(method="NLOPT_LN_PRAXIS").set_name("NLOPT_LN_PRAXIS", register=True)
NLOPT_GN_DIRECT = NonObjectOptimizer(method="NLOPT_GN_DIRECT").set_name("NLOPT_GN_DIRECT", register=True)
NLOPT_GN_DIRECT_L = NonObjectOptimizer(method="NLOPT_GN_DIRECT_L").set_name(
    "NLOPT_GN_DIRECT_L", register=True
)
NLOPT_GN_CRS2_LM = NonObjectOptimizer(method="NLOPT_GN_CRS2_LM").set_name("NLOPT_GN_CRS2_LM", register=True)
NLOPT_GN_AGS = NonObjectOptimizer(method="NLOPT_GN_AGS").set_name("NLOPT_GN_AGS", register=True)
NLOPT_GN_ISRES = NonObjectOptimizer(method="NLOPT_GN_ISRES").set_name("NLOPT_GN_ISRES", register=True)
NLOPT_GN_ESCH = NonObjectOptimizer(method="NLOPT_GN_ESCH").set_name("NLOPT_GN_ESCH", register=True)
NLOPT_LN_COBYLA = NonObjectOptimizer(method="NLOPT_LN_COBYLA").set_name("NLOPT_LN_COBYLA", register=True)
NLOPT_LN_BOBYQA = NonObjectOptimizer(method="NLOPT_LN_BOBYQA").set_name("NLOPT_LN_BOBYQA", register=True)
NLOPT_LN_NEWUOA_BOUND = NonObjectOptimizer(method="NLOPT_LN_NEWUOA_BOUND").set_name(
    "NLOPT_LN_NEWUOA_BOUND", register=True
)
NLOPT_LN_NELDERMEAD = NonObjectOptimizer(method="NLOPT_LN_NELDERMEAD").set_name(
    "NLOPT_LN_NELDERMEAD", register=True
)


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
            self._initial_seed = self._rng.randint(2**30)
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
        problem = _create_pymoo_problem(weakself, objective_function)
        if weakself.algorithm == "CMAES":
            from pymoo.algorithms.soo.nonconvex.cmaes import CMAES

            algorithm = CMAES(x0=np.random.random(problem.n_var), maxfevals=weakself.budget)
        elif weakself.algorithm == "BIPOP":
            from pymoo.algorithms.soo.nonconvex.cmaes import CMAES

            algorithm = CMAES(
                x0=np.random.random(problem.n_var),
                sigma=0.5,
                restarts=2,
                maxfevals=weakself.budget,
                tolfun=1e-6,
                tolx=1e-6,
                restart_from_best=True,
                bipop=True,
            )
        else:
            algorithm = get_pymoo_algorithm(weakself.algorithm)
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
            self._initial_seed = self._rng.randint(2**30)
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


PymooCMAES = Pymoo(algorithm="CMAES").set_name("PymooCMAES", register=True)
PymooBIPOP = Pymoo(algorithm="BIPOP").set_name("PymooBIPOP", register=True)
PymooNSGA2 = Pymoo(algorithm="nsga2").set_name("PymooNSGA2", register=True)
PymooBatchNSGA2 = PymooBatch(algorithm="nsga2").set_name("PymooBatchNSGA2", register=False)
