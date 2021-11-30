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
import pybobyqa
from ax import optimize as axoptimize
import cma
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
        #assert method in ["Nelder-Mead", "COBYLA", "SLSQP", "Powell", "BOBYQA", "AX"], f"Unknown method '{method}'"
        #assert method in ["SMAC2", "SMAC", "Nelder-Mead", "COBYLA", "SLSQP", "Powell"], f"Unknown method '{method}'"
        self.method = method
        self.random_restart = random_restart
        self._normalizer = p.helpers.Normalizer(self.parametrization)
        assert method in [
            "CmaFmin2",
            "SMAC2",
            "SMAC",
            "AX",
            "Lamcts",
            "Nelder-Mead",
            "COBYLA",
            "SLSQP",
            "Powell",
        ], f"Unknown method '{method}'"
        self.method = method
        self.random_restart = random_restart
        # The following line rescales to [0, 1] if fully bounded.

        if method == "CmaFmin2":
            normalizer = p.helpers.Normalizer(self.parametrization)
            if normalizer.fully_bounded:
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
        best_x: np.ndarray = self.current_bests["average"].x
        if self.initial_guess is not None:
            best_x = np.array(weakself.initial_guess, copy=True)  # copy, just to make sure it is not modified
        remaining = budget - self._num_ask
        def ax_obj(p):
            data = [p["x" + str(i)] for i in range(self.dimension)]
            data = self._normalizer.backward(np.asarray(data, dtype=np.float))
            return objective_function(data)
        while remaining > 0:  # try to restart if budget is not elapsed
            print(f"Iteration with remaining={remaining}")
            options: tp.Dict[str, tp.Any] = {} if self.budget is None else {"maxiter": remaining}
            if weakself.method == "BOBYQA":
                res = pybobyqa.solve(objective_function, best_x, maxfun=budget, do_logging=False)
                if res.f < best_res:
                    best_res = res.f
                    best_x = res.x
            elif weakself.method == "AX":
                parameters = [{"name": "x"+str(i), "type":"range", "bounds":[0., 1.]} for i in range(weakself.dimension)]
                best_parameters, best_values, experiment, model = axoptimize(
                    parameters,
                    evaluation_function = ax_obj,
                    minimize=True,
                    total_trials = budget)
                best_x = [p["x" + str(i)] for i in range(weakself.dimension)]
                best_x = weakself._normalizer.backward(np.asarray(best_x, dtype=np.float))
            elif weakself.method == "SMAC2":
                from ConfigSpace.hyperparameters import (
                    UniformFloatHyperparameter,
                )  # noqa  # pylint: disable=unused-import

                # Import ConfigSpace and different types of parameters
                from smac.configspace import ConfigurationSpace  # noqa  # pylint: disable=unused-import
                from smac.facade.smac_hpo_facade import SMAC4HPO  # noqa  # pylint: disable=unused-import

                # Import SMAC-utilities
                from smac.scenario.scenario import Scenario  # noqa  # pylint: disable=unused-import
                import threading
                import os
                import time
                from pathlib import Path
                the_date = str(time.time())
                feed = "/tmp/smac_feed" + the_date + ".txt"
                fed = "/tmp/smac_fed" + the_date + ".txt"
                def dummy_function():
                    for u in range(remaining):
                        print(f"side thread waiting for request... ({u}/{self.budget})")
                        while (not Path(feed).is_file()) or os.stat(feed).st_size == 0:
                            time.sleep(0.1)
                        time.sleep(0.1)
                        print("side thread happy to work on a request...")
                        data = np.loadtxt(feed)
                        os.remove(feed)
                        print("side thread happy to really work on a request...")
                        res = objective_function(data)
                        print("side thread happy to forward the result of a request...")
                        f = open(fed, "w")
                        f.write(str(res))
                        f.close()
                    return
                thread = threading.Thread(target=dummy_function)
                thread.start()


                print(f"start SMAC2 optimization with budget {budget} in dimension {self.dimension}")
                cs = ConfigurationSpace()
                cs.add_hyperparameters(
                    [
                        UniformFloatHyperparameter(f"x{i}", 0.0, 1.0, default_value=0.0)
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
                    p = [p[f"x{i}"] for i in range(len(p.keys()))]
                    data = weakself._normalizer.backward(np.asarray(p, dtype=np.float))
                    print(f"converted to {data}")
                    if Path(fed).is_file():
                        os.remove(fed)
                    np.savetxt(feed, data)
                    while (not Path(fed).is_file()) or os.stat(fed).st_size == 0:
                        time.sleep(0.1)
                    time.sleep(0.1)
                    f = open(fed, "r")
                    res = np.float(f.read())
                    f.close()
                    print(f"SMAC2 will receive {res}")
                    return res
                smac = SMAC4HPO(scenario=scenario, rng=self._rng.randint(5000), tae_runner=smac2_obj)
                res = smac.optimize()
                best_x = [res[f"x{k}"] for k in range(len(res.keys()))]
                best_x = weakself._normalizer.backward(np.asarray(best_x, dtype=np.float))
                print("end SMAC optimization")
                thread.join()
                weakself._num_ask = budget

            elif weakself.method == "SMAC":
                import smac  # noqa  # pylint: disable=unused-import
                import scipy.optimize  # noqa  # pylint: disable=unused-import
                from smac.facade.func_facade import fmin_smac  # noqa  # pylint: disable=unused-import

                import threading
                import os
                import time
                from pathlib import Path
                the_date = str(time.time())
                feed = "/tmp/smac_feed" + the_date + ".txt"
                fed = "/tmp/smac_fed" + the_date + ".txt"
                def dummy_function():
                    for u in range(remaining):
                        print(f"side thread waiting for request... ({u}/{self.budget})")
                        while (not Path(feed).is_file()) or os.stat(feed).st_size == 0:
                            time.sleep(0.1)
                        time.sleep(0.1)
                        print("side thread happy to work on a request...")
                        data = np.loadtxt(feed)
                        os.remove(feed)
                        print("side thread happy to really work on a request...")
                        res = objective_function(data)
                        print("side thread happy to forward the result of a request...")
                        f = open(fed, "w")
                        f.write(str(res))
                        f.close()
                    return
                thread = threading.Thread(target=dummy_function)
                thread.start()

                def smac_obj(p):
                    print(f"SMAC proposes {p}")
                    data = weakself._normalizer.backward(np.asarray([p[i] for i in range(len(p))], dtype=np.float))
                    print(f"converted to {data}")
                    if Path(fed).is_file():
                        os.remove(fed)
                    np.savetxt(feed, data)
                    while (not Path(fed).is_file()) or os.stat(fed).st_size == 0:
                        time.sleep(0.1)
                    time.sleep(0.1)
                    f = open(fed, "r")
                    res = np.float(f.read())
                    f.close()
                    print(f"SMAC will receive {res}")
                    return res

                print(f"start SMAC optimization with budget {budget} in dimension {weakself.dimension}")
                assert budget is not None
                x, cost, _ = fmin_smac(
                    #func=lambda x: sum([(x_ - 1.234)**2  for x_ in x]),
                    func=smac_obj,
                    x0=[0.0] * weakself.dimension,
                    bounds=[(0., 1.)] * weakself.dimension,
                    maxfun=remaining,
                    rng=weakself._rng.randint(5000),
                )  # Passing a seed makes fmin_smac determistic
                print("end SMAC optimization")
                thread.join()
                weakself._num_ask = budget

                if cost < best_res:
                    best_res = cost
                    best_x = weakself._normalizer.backward(np.asarray(x, dtype=np.float))
            else:
                res = scipyoptimize.minimize(
                    objective_function,
                    best_x if not weakself.random_restart else weakself._rng.normal(0.0, 1.0, weakself.dimension),
                    method=weakself.method,
#        best_x: np.ndarray = weakself.current_bests["average"].x  # np.zeros(self.dimension)
#        if weakself.initial_guess is not None:
#            best_x = np.array(weakself.initial_guess, copy=True)  # copy, just to make sure it is not modified
#        remaining: float = budget - weakself._num_ask
#        while remaining > 0:  # try to restart if budget is not elapsed
#            options: tp.Dict[str, tp.Any] = {} if weakself.budget is None else {"maxiter": remaining}
#            # options: tp.Dict[str, tp.Any] = {} if self.budget is None else {"maxiter": remaining}
#            if weakself.method == "CmaFmin2":
#
#                def cma_objective_function(data):
#                    # Hopefully the line below does nothing if unbounded and rescales from [0, 1] if bounded.
#                    if weakself._normalizer is not None:
#                        data = weakself._normalizer.backward(np.asarray(data, dtype=np.float32))
#                    return objective_function(data)
#
#                # cma.fmin2(objective_function, [0.0] * self.dimension, [1.0] * self.dimension, remaining)
#                x0 = 0.5 * np.ones(weakself.dimension)
#                num_calls = 0
#                while budget - num_calls > 0:
#                    options = {"maxfevals": budget - num_calls, "verbose": -9}
#                    if weakself._normalizer is not None:
#                        # Tell CMA to work in [0, 1].
#                        options["bounds"] = [0.0, 1.0]
#                    res = cma.fmin(
#                        cma_objective_function,
#                        x0=x0,
#                        sigma0=0.2,
#                        options=options,
#                        restarts=9,
#                    )
#                    x0 = 0.5 + np.random.uniform() * np.random.uniform(
#                        low=-0.5, high=0.5, size=weakself.dimension
#                    )
#                    if res[1] < best_res:
#                        best_res = res[1]
#                        best_x = res[0]
#                        if weakself._normalizer is not None:
#                            best_x = weakself._normalizer.backward(np.asarray(best_x, dtype=np.float32))
#                    num_calls += res[2]
#            else:
#                res = scipyoptimize.minimize(
#                    objective_function,
#                    best_x
#                    if not weakself.random_restart
#                    else weakself._rng.normal(0.0, 1.0, weakself.dimension),
#                    method=weakself.method,
                    options=options,
                    tol=0,
                )
                if res.fun < best_res:
                    best_res = res.fun
                    best_x = res.x
            remaining = budget - weakself._num_ask
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
Powell = NonObjectOptimizer(method="Powell").set_name("Powell", register=True)
RPowell = NonObjectOptimizer(method="Powell", random_restart=True).set_name("RPowell", register=True)
Cobyla = NonObjectOptimizer(method="COBYLA").set_name("Cobyla", register=True)
RCobyla = NonObjectOptimizer(method="COBYLA", random_restart=True).set_name("RCobyla", register=True)
SQP = NonObjectOptimizer(method="SLSQP").set_name("SQP", register=True)
SLSQP = SQP  # Just so that people who are familiar with SLSQP naming are not lost.
RSQP = NonObjectOptimizer(method="SLSQP", random_restart=True).set_name("RSQP", register=True)
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


from .lamcts.MCTS import lamcts_minimize

class _LamctsMinimizeBase(recaster.SequentialRecastOptimizer):

    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
        *,
        method: str = "Nelder-Mead",
        random_restart: bool = False,
        device: str = 'cpu',
    ) -> None:
        super().__init__(parametrization, budget=budget, num_workers=num_workers)
        self.multirun = 1  # work in progress
        self.initial_guess: tp.Optional[tp.ArrayLike] = None
        # configuration
        assert method in ["Nelder-Mead", "COBYLA", "SLSQP", "Powell"], f"Unknown method '{method}'"
        self.method = method
        self.random_restart = random_restart
        self.device = device

    def _internal_tell_not_asked(self, candidate: p.Parameter, value: float) -> None:
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
            random_restart=self.random_restart)
        subinstance.archive = self.archive
        subinstance.current_bests = self.current_bests
        return subinstance._optimization_function

    def _optimization_function(self, objective_function: tp.Callable[[tp.ArrayLike], float]) -> tp.ArrayLike:
        # pylint:disable=unused-argument
        budget = np.inf if self.budget is None else self.budget
        best_res = np.inf
        best_x: np.ndarray = weakself.current_bests["average"].x  # np.zeros(self.dimension)
        if weakself.initial_guess is not None:
            best_x = np.array(self.initial_guess, copy=True)  # copy, just to make sure it is not modified
        remaining = budget - self._num_ask
        while remaining > 0:  # try to restart if budget is not elapsed
            options: Dict[str, int] = {} if self.budget is None else {"maxiter": remaining}
            res = lamcts_minimize(
                func=objective_function,
                dims=self.parametrization.dimension,
                budget=self.budget,
                device=self.device,
#                best_x if not self.random_restart else self._rng.normal(0.0, 1.0, self.dimension),
#                method=self.method,
#                options=options,
#                tol=0,
            )
#def lamcts_minimize(func, dims, budget, lb=None, ub=None):
            if res.fun < best_res:
                best_res = res.fun
                best_x = res.x
            remaining = budget - self._num_ask
        return best_x


class LamctsOptimizer(base.ConfiguredOptimizer):
    """Wrapper over Lamcts optimizer implementations, in standard ask and tell format.
Sequential Quadratic Programming. Inside Nevergrad, this code is in https://github.com/facebookresearch/nevergrad/blob/master/nevergrad/optimization/optimizerlib.py; this is actually an import from scipy-optimize. It is very powerful e.g. in continuous noisy optimization. It is based on approximating the objective function by quadratic models.

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
    def __init__(
        self,
        *,
        random_restart: bool = False,
        device: str = 'cpu'
    ) -> None:
        super().__init__(_LamctsMinimizeBase, locals())




PymooBatchNSGA2 = PymooBatch(algorithm="nsga2").set_name("PymooBatchNSGA2", register=False)
