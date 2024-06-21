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
        # assert method in ["Nelder-Mead", "COBYLA", "SLSQP", "Powell", "BOBYQA", "AX"], f"Unknown method '{method}'"
        # assert method in ["SMAC3", "SMAC", "Nelder-Mead", "COBYLA", "SLSQP", "Powell"], f"Unknown method '{method}'"
        self.method = method
        self.random_restart = random_restart
        # self._normalizer = p.helpers.Normalizer(self.parametrization)
        assert (
            method
            in [
                "CmaFmin2",
                "gomea",
                "gomeablock",
                "gomeatree",
                "SMAC3",
                "BFGS",
                "RBFGS",
                "LBFGSB",
                "L-BFGS-B",
                "SMAC",
                "AX",
                "Lamcts",
                "Nelder-Mead",
                "COBYLA",
                "BOBYQA",
                "SLSQP",
                "pysot",
                "negpysot",
                "Powell",
            ]
            or "NLOPT" in method
            or "DS" in method
            or "BFGS" in method
        ), f"Unknown method '{method}'"
        if (
            method == "CmaFmin2"
            or "NLOPT" in method
            or "AX" in method
            or "BOBYQA" in method
            or "pysot" in method
            or "SMAC" in method
        ):
            normalizer = p.helpers.Normalizer(self.parametrization)
            #            if normalizer.fully_bounded or method == "AX" or "pysot" == method or "SMAC" in method:
            #                self._normalizer = normalizer
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
        best_x: np.ndarray = weakself.current_bests["average"].x
        if weakself.initial_guess is not None:
            best_x = np.array(weakself.initial_guess, copy=True)  # copy, just to make sure it is not modified

        remaining: float = budget - weakself._num_ask

        def ax_obj(p):
            data = [p["x" + str(i)] for i in range(weakself.dimension)]  # type: ignore
            if weakself._normalizer:
                data = weakself._normalizer.backward(np.asarray(data, dtype=np.float64))
            return objective_function(data)

        while remaining > 0:  # try to restart if budget is not elapsed
            # print(f"Iteration with remaining={remaining}")
            options: tp.Dict[str, tp.Any] = {} if weakself.budget is None else {"maxiter": remaining}
            if weakself.method == "BOBYQA" or (weakself.method == "CmaFmin2" and weakself.dimension == 1):
                import pybobyqa  # type: ignore

                res = pybobyqa.solve(objective_function, best_x, maxfun=budget, do_logging=False)
                if res.f < best_res:
                    best_res = res.f
                    best_x = res.x
            elif weakself.method[:2] == "DS":
                import directsearch  # type: ignore

                dict_solvers = {
                    "base": directsearch.solve_directsearch,
                    "proba": directsearch.solve_probabilistic_directsearch,
                    "subspace": directsearch.solve_subspace_directsearch,
                    "3p": directsearch.solve_stp,
                }
                solve = dict_solvers[weakself.method[2:]]
                best_x = solve(objective_function, x0=best_x, maxevals=budget).x
                if weakself._normalizer is not None:
                    best_x = weakself._normalizer.backward(np.asarray(best_x, dtype=np.float32))
            elif weakself.method[:3] == "PDS":
                import directsearch  # type: ignore

                solve = directsearch.solve_probabilistic_directsearch
                DSseed = int(weakself.method[3:])
                best_x = solve(
                    objective_function,
                    x0=best_x,
                    maxevals=budget,
                    gamma_inc=1.0 + np.random.RandomState(DSseed).rand() * 3.0,
                    gamma_dec=np.random.RandomState(DSseed + 42).rand(),
                ).x
                if weakself._normalizer is not None:
                    best_x = weakself._normalizer.backward(np.asarray(best_x, dtype=np.float32))
            elif weakself.method == "AX":
                from ax import optimize as axoptimize  # type: ignore

                parameters = [
                    {"name": "x" + str(i), "type": "range", "bounds": [0.0, 1.0]}
                    for i in range(weakself.dimension)
                ]
                best_parameters, _best_values, _experiment, _model = axoptimize(
                    parameters, evaluation_function=ax_obj, minimize=True, total_trials=budget
                )
                best_x = np.array([float(best_parameters["x" + str(i)]) for i in range(weakself.dimension)])
                best_x = weakself._normalizer.backward(np.asarray(best_x, dtype=float))
            # options: tp.Dict[str, tp.Any] = {} if weakself.budget is None else {"maxiter": remaining}
            elif weakself.method[:5] == "NLOPT":
                # This is NLOPT, used as in the PCSE simulator notebook.
                # ( https://github.com/ajwdewit/pcse_notebooks ).
                import nlopt  # type: ignore

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
            elif "pysot" in weakself.method:
                from poap.controller import BasicWorkerThread, ThreadController  # type: ignore

                from pySOT.experimental_design import SymmetricLatinHypercube  # type: ignore
                from pySOT.optimization_problems import OptimizationProblem  # type: ignore

                # from pySOT.strategy import SRBFStrategy
                from pySOT.strategy import DYCORSStrategy  # type: ignore
                from pySOT.surrogate import CubicKernel, LinearTail, RBFInterpolant  # type: ignore

                class LocalOptimizationProblem(OptimizationProblem):
                    def eval(self, data):
                        if weakself._normalizer is not None:
                            data = weakself._normalizer.backward(np.asarray(data, dtype=np.float32))
                        val = (
                            float(objective_function(data))
                            if "negpysot" not in weakself.method
                            else -float(objective_function(data))
                        )
                        return val

                dim = weakself.dimension
                opt_prob = LocalOptimizationProblem()
                opt_prob.dim = dim
                opt_prob.lb = np.array([0.0] * dim)
                opt_prob.ub = np.array([1.0] * dim)
                opt_prob.int_var = []
                opt_prob.cont_var = np.array(range(dim))

                rbf = RBFInterpolant(
                    dim=opt_prob.dim,
                    lb=opt_prob.lb,
                    ub=opt_prob.ub,
                    kernel=CubicKernel(),
                    tail=LinearTail(opt_prob.dim),
                )
                slhd = SymmetricLatinHypercube(dim=opt_prob.dim, num_pts=2 * (opt_prob.dim + 1))
                controller = ThreadController()
                # controller.strategy = SRBFStrategy(
                #    max_evals=budget, opt_prob=opt_prob, exp_design=slhd, surrogate=rbf, asynchronous=True
                # )
                controller.strategy = DYCORSStrategy(
                    opt_prob=opt_prob, exp_design=slhd, surrogate=rbf, max_evals=budget, asynchronous=True
                )
                worker = BasicWorkerThread(controller, opt_prob.eval)
                controller.launch_worker(worker)

                result = controller.run()

                best_res = result.value
                best_x = result.params[0]

            elif weakself.method == "SMAC3":

                # Import ConfigSpace and different types of parameters
                # from smac.configspace import ConfigurationSpace  # type: ignore  # noqa  # pylint: disable=unused-import
                # from smac.configspace import UniformFloatHyperparameter  # type: ignore
                # from smac.facade.smac_hpo_facade import SMAC4HPO  # type: ignore  # noqa  # pylint: disable=unused-import

                from ConfigSpace import Configuration, ConfigurationSpace, UniformFloatHyperparameter
                from smac import HyperparameterOptimizationFacade, Scenario

                # Import SMAC-utilities
                import threading
                import os
                import time
                from pathlib import Path

                the_date = str(time.time()) + "_" + str(np.random.rand())
                tag = str(np.random.rand())
                feed = "/tmp/smac_feed" + the_date + ".txt"
                fed = "/tmp/smac_fed" + the_date + ".txt"

                def dummy_function():
                    for _ in range(remaining):
                        # print(f"side thread waiting for request... ({u}/{weakself.budget})")
                        while (not Path(feed).is_file()) or os.stat(feed).st_size == 0:
                            time.sleep(0.1)
                        time.sleep(0.1)
                        # print("side thread happy to work on a request...")
                        data = np.loadtxt(feed)
                        os.remove(feed)
                        # print("side thread happy to really work on a request...")
                        res = objective_function(data)
                        # print("side thread happy to forward the result of a request...")
                        f = open(fed, "w")
                        f.write(str(res))
                        f.close()
                    return

                thread = threading.Thread(target=dummy_function)
                thread.start()

                # print(f"start SMAC3 optimization with budget {budget} in dimension {weakself.dimension}")
                cs = ConfigurationSpace()
                cs.add_hyperparameters(
                    [
                        UniformFloatHyperparameter(f"x{tag}{i}", 0.0, 1.0, default_value=0.0)
                        for i in range(weakself.dimension)
                    ]
                )

                def smac2_obj(p, seed: int = 0):
                    # print(f"SMAC3 proposes {p} {type(p)}")
                    pdata = [p[f"x{tag}{i}"] for i in range(len(p.keys()))]
                    data = weakself._normalizer.backward(np.asarray(pdata, dtype=float))
                    # print(f"converted to {data}")
                    if Path(fed).is_file():
                        os.remove(fed)
                    np.savetxt(feed, data)
                    while (not Path(fed).is_file()) or os.stat(fed).st_size == 0:
                        time.sleep(0.1)
                    time.sleep(0.1)
                    f = open(fed, "r")
                    res = float(f.read())
                    f.close()
                    # print(f"SMAC3 will receive {res}")
                    return res

                # scenario = Scenario({'cs': cs, 'run_obj': smac2_obj, 'runcount-limit': remaining, 'deterministic': True})
                scenario = Scenario(cs, deterministic=True, n_trials=int(remaining))

                smac = HyperparameterOptimizationFacade(scenario, smac2_obj)
                res = smac.optimize()
                best_x = np.array([res[f"x{tag}{k}"] for k in range(len(res.keys()))])
                best_x = weakself._normalizer.backward(np.asarray(best_x, dtype=float))
                # print(f"end SMAC optimization {best_x}")
                thread.join()
                weakself._num_ask = budget

            #            elif weakself.method == "SMAC":
            #                import smac  # noqa  # pylint: disable=unused-import
            #                import scipy.optimize  # noqa  # pylint: disable=unused-import
            #                from smac.facade.func_facade import fmin_smac  # noqa  # pylint: disable=unused-import
            #
            #                import threading
            #                import os
            #                import time
            #                from pathlib import Path
            #
            #                the_date = str(time.time())
            #                feed = "/tmp/smac_feed" + the_date + ".txt"
            #                fed = "/tmp/smac_fed" + the_date + ".txt"
            #
            #                def dummy_function():
            #                    for u in range(remaining):
            #                        print(f"side thread waiting for request... ({u}/{weakself.budget})")
            #                        while (not Path(feed).is_file()) or os.stat(feed).st_size == 0:
            #                            time.sleep(0.1)
            #                        time.sleep(0.1)
            #                        print("side thread happy to work on a request...")
            #                        data = np.loadtxt(feed)
            #                        os.remove(feed)
            #                        print("side thread happy to really work on a request...")
            #                        res = objective_function(data)
            #                        print("side thread happy to forward the result of a request...")
            #                        f = open(fed, "w")
            #                        f.write(str(res))
            #                        f.close()
            #                    return
            #
            #                thread = threading.Thread(target=dummy_function)
            #                thread.start()
            #
            #                def smac_obj(p):
            #                    print(f"SMAC proposes {p}")
            #                    data = weakself._normalizer.backward(
            #                        np.asarray([p[i] for i in range(len(p))], dtype=np.float)
            #                    )
            #                    print(f"converted to {data}")
            #                    if Path(fed).is_file():
            #                        os.remove(fed)
            #                    np.savetxt(feed, data)
            #                    while (not Path(fed).is_file()) or os.stat(fed).st_size == 0:
            #                        time.sleep(0.1)
            #                    time.sleep(0.1)
            #                    f = open(fed, "r")
            #                    res = np.float(f.read())
            #                    f.close()
            #                    print(f"SMAC will receive {res}")
            #                    return res
            #
            #                print(f"start SMAC optimization with budget {budget} in dimension {weakself.dimension}")
            #                assert budget is not None
            #                x, cost, _ = fmin_smac(
            #                    # func=lambda x: sum([(x_ - 1.234)**2  for x_ in x]),
            #                    func=smac_obj,
            #                    x0=[0.0] * weakself.dimension,
            #                    bounds=[(0.0, 1.0)] * weakself.dimension,
            #                    maxfun=remaining,
            #                    rng=weakself._rng.randint(5000),
            #                )  # Passing a seed makes fmin_smac determistic
            #                print("end SMAC optimization")
            #                thread.join()
            #                weakself._num_ask = budget
            #
            #                if cost < best_res:
            #                    best_res = cost
            #                    best_x = weakself._normalizer.backward(np.asarray(x, dtype=float))
            #

            #            elif "gomea" in weakself.method:
            #                import gomea
            #
            #                class gomea_function(gomea.fitness.BBOFitnessFunctionRealValued):
            #                    def objective_function(self, objective_index, data):  # type: ignore
            #                        if weakself._normalizer is not None:
            #                            data = weakself._normalizer.backward(np.asarray(data, dtype=np.float32))
            #                        return objective_function(data)
            #
            #                gomea_f = gomea_function(weakself.dimension)
            #                lm = {
            #                    "gomea": gomea.linkage.Univariate(),
            #                    "gomeablock": gomea.linkage.BlockMarginalProduct(2),
            #                    "gomeatree": gomea.linkage.LinkageTree("NMI".encode(), True, 0),
            #                }[weakself.method]
            #                rvgom = gomea.RealValuedGOMEA(
            #                    fitness=gomea_f,
            #                    linkage_model=lm,
            #                    lower_init_range=0.0,
            #                    upper_init_range=1.0,
            #                    max_number_of_evaluations=budget,
            #                )
            #                rvgom.run()
            #                best_x = gomea_f.best_x

            elif weakself.method == "CmaFmin2" and weakself.dimension > 1:
                import cma  # type: ignore

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
                    (
                        best_x
                        if not weakself.random_restart
                        else weakself._rng.normal(0.0, 1.0, weakself.dimension)
                    ),
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


AX = NonObjectOptimizer(method="AX").set_name("AX", register=True)
BOBYQA = NonObjectOptimizer(method="BOBYQA").set_name("BOBYQA", register=True)
NelderMead = NonObjectOptimizer(method="Nelder-Mead").set_name("NelderMead", register=True)
CmaFmin2 = NonObjectOptimizer(method="CmaFmin2").set_name("CmaFmin2", register=True)
# GOMEA = NonObjectOptimizer(method="gomea").set_name("GOMEA", register=True)
# GOMEABlock = NonObjectOptimizer(method="gomeablock").set_name("GOMEABlock", register=True)
# GOMEATree = NonObjectOptimizer(method="gomeatree").set_name("GOMEATree", register=True)
# NLOPT = NonObjectOptimizer(method="NLOPT").set_name("NLOPT", register=True)
Powell = NonObjectOptimizer(method="Powell").set_name("Powell", register=True)
RPowell = NonObjectOptimizer(method="Powell", random_restart=True).set_name("RPowell", register=True)
BFGS = NonObjectOptimizer(method="BFGS", random_restart=False).set_name("BFGS", register=True)
RBFGS = NonObjectOptimizer(method="BFGS", random_restart=True).set_name("RBFGS", register=True)
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
# AX = NonObjectOptimizer(method="AX").set_name("AX", register=True)
# BOBYQA = NonObjectOptimizer(method="BOBYQA").set_name("BOBYQA", register=True)
# SMAC = NonObjectOptimizer(method="SMAC").set_name("SMAC", register=True)
SMAC3 = NonObjectOptimizer(method="SMAC3").set_name("SMAC3", register=True)


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


##Not yet included, coming.
# from .lamcts.MCTS import lamcts_minimize
#
#
# class _LamctsMinimizeBase(recaster.SequentialRecastOptimizer):
#    def __init__(
#        self,
#        parametrization: IntOrParameter,
#        budget: tp.Optional[int] = None,
#        num_workers: int = 1,
#        *,
#        method: str = "Nelder-Mead",
#        random_restart: bool = False,
#        device: str = "cpu",
#    ) -> None:
#        super().__init__(parametrization, budget=budget, num_workers=num_workers)
#        self.multirun = 1  # work in progress
#        normalizer = p.helpers.Normalizer(self.parametrization)
#        self._normalizer = normalizer
#
#        self.initial_guess: tp.Optional[tp.ArrayLike] = None
#        # configuration
#        assert method in ["Nelder-Mead", "COBYLA", "SLSQP", "Powell"], f"Unknown method '{method}'"
#        self.method = method
#        self.random_restart = random_restart
#        self.device = device
#
#    def _internal_tell_not_asked(self, candidate: p.Parameter, value: float) -> None:
#        """Called whenever calling "tell" on a candidate that was not "asked".
#        Defaults to the standard tell pipeline.
#        """  # We do not do anything; this just updates the current best.
#
#    def get_optimization_function(self) -> tp.Callable[[tp.Callable[[tp.ArrayLike], float]], tp.ArrayLike]:
#        # create a different sub-instance, so that the current instance is not referenced by the thread
#        # (consequence: do not create a thread at initialization, or we get a thread explosion)
#        subinstance = self.__class__(
#            parametrization=self.parametrization,
#            budget=self.budget,
#            num_workers=self.num_workers,
#            method=self.method,
#            random_restart=self.random_restart,
#        )
#        subinstance.archive = self.archive
#        subinstance.current_bests = self.current_bests
#        return subinstance._optimization_function
#
#    def _optimization_function(self, objective_function: tp.Callable[[tp.ArrayLike], float]) -> tp.ArrayLike:
#        # pylint:disable=unused-argument
#        budget = np.inf if self.budget is None else self.budget
#        best_res = np.inf
#        best_x: np.ndarray = self.current_bests["average"].x  # np.zeros(self.dimension)
#        if self.initial_guess is not None:
#            best_x = np.array(self.initial_guess, copy=True)  # copy, just to make sure it is not modified
#        remaining = budget - self._num_ask
#        while remaining > 0:  # try to restart if budget is not elapsed
#            options: Dict[str, int] = {} if self.budget is None else {"maxiter": remaining}
#
#            def lamcts_obj(data):
#                # print("transform", data)
#                data = (data + 1.0) / 2.0
#                data = self._normalizer.backward(np.asarray(data, dtype=np.float))
#                return objective_function(data)
#
#            res = lamcts_minimize(
#                # func=objective_function,
#                func=lamcts_obj,
#                dims=self.parametrization.dimension,
#                budget=self.budget,
#                device=self.device,
#                #                best_x if not self.random_restart else self._rng.normal(0.0, 1.0, self.dimension),
#                #                method=self.method,
#                #                options=options,
#                #                tol=0,
#            )
#            # def lamcts_minimize(func, dims, budget, lb=None, ub=None):
#            if res.fun < best_res:
#                best_res = res.fun
#                best_x = res.x
#                best_x = 2.0 * weakself._normalizer.backward(np.asarray(best_x, dtype=np.float32)) - 1.0
#            remaining = budget - self._num_ask
#        return best_x
#
#
# class LamctsOptimizer(base.ConfiguredOptimizer):
#    """Wrapper over Lamcts optimizer implementations, in standard ask and tell format.
#    Sequential Quadratic Programming. Inside Nevergrad, this code is in https://github.com/facebookresearch/nevergrad/blob/master/nevergrad/optimization/optimizerlib.py; this is actually an import from scipy-optimize. It is very powerful e.g. in continuous noisy optimization. It is based on approximating the objective function by quadratic models.
#
#        Parameters
#        ----------
#        method: str
#            Name of the method to use among:
#
#            - Nelder-Mead
#            - COBYLA
#            - SQP (or SLSQP): very powerful e.g. in continuous noisy optimization. It is based on
#              approximating the objective function by quadratic models.
#            - Powell
#        random_restart: bool
#            whether to restart at a random point if the optimizer converged but the budget is not entirely
#            spent yet (otherwise, restarts from best point)
#
#        Note
#        ----
#        These optimizers do not support asking several candidates in a row
#    """
#
#    recast = True
#    no_parallelization = True
#
#    # pylint: disable=unused-argument
#    def __init__(self, *, random_restart: bool = False, device: str = "cpu") -> None:
#        super().__init__(_LamctsMinimizeBase, locals())
#

PymooBatchNSGA2 = PymooBatch(algorithm="nsga2").set_name("PymooBatchNSGA2", register=False)
pysot = NonObjectOptimizer(method="pysot").set_name("pysot", register=True)

DSbase = NonObjectOptimizer(method="DSbase").set_name("DSbase", register=True)
DS3p = NonObjectOptimizer(method="DS3p").set_name("DS3p", register=True)
DSsubspace = NonObjectOptimizer(method="DSsubspace").set_name("DSsubspace", register=True)
DSproba = NonObjectOptimizer(method="DSproba").set_name("DSproba", register=True)
# DSproba2 = NonObjectOptimizer(method="PDS2").set_name("DSproba2", register=True)
# DSproba3 = NonObjectOptimizer(method="PDS3").set_name("DSproba3", register=True)
# DSproba4 = NonObjectOptimizer(method="PDS4").set_name("DSproba4", register=True)
# DSproba5 = NonObjectOptimizer(method="PDS5").set_name("DSproba5", register=True)
# DSproba6 = NonObjectOptimizer(method="PDS6").set_name("DSproba6", register=True)
# DSproba7 = NonObjectOptimizer(method="PDS7").set_name("DSproba7", register=True)
# DSproba8 = NonObjectOptimizer(method="PDS8").set_name("DSproba8", register=True)
# DSproba9 = NonObjectOptimizer(method="PDS9").set_name("DSproba9", register=True)
