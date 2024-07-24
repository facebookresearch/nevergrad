# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from pathlib import Path
import logging
import os
import numpy as np
import nevergrad as ng
import nevergrad.common.typing as tp
from . import optimizerlib
from . import callbacks


# pylint: disable=unused-argument
def _func(x: tp.Any, y: tp.Any, blublu: str, array: tp.Any, multiobjective: bool = False) -> tp.Loss:
    return 12 if not multiobjective else [12, 12]


def test_log_parameters(tmp_path: Path) -> None:
    filepath = tmp_path / "logs.txt"
    cases = [0, np.int_(1), np.float64(2.0), np.nan, float("inf"), np.inf]
    instrum = ng.p.Instrumentation(
        ng.ops.mutations.Translation()(ng.p.Array(shape=(1,))),
        ng.p.Scalar(),
        blublu=ng.p.Choice(cases),
        array=ng.p.Array(shape=(3, 2)),
    )
    optimizer = optimizerlib.NoisyOnePlusOne(parametrization=instrum, budget=32)
    optimizer.register_callback("tell", ng.callbacks.ParametersLogger(filepath, append=False))
    optimizer.minimize(_func, verbosity=2)
    # pickling
    logger = callbacks.ParametersLogger(filepath)
    logs = logger.load_flattened()
    assert len(logs) == 32
    assert isinstance(logs[-1]["1"], float)
    assert len(logs[-1]) == 38
    logs = logger.load_flattened(max_list_elements=2)
    assert len(logs[-1]) == 34
    # deletion
    logger = callbacks.ParametersLogger(filepath, append=False)
    assert not logger.load()


def test_multiobjective_log_parameters(tmp_path: Path) -> None:
    filepath = tmp_path / "logs.txt"
    instrum = ng.p.Instrumentation(
        None, 2.0, blublu="blublu", array=ng.p.Array(shape=(3, 2)), multiobjective=True
    )
    optimizer = optimizerlib.OnePlusOne(parametrization=instrum, budget=2)
    optimizer.register_callback("tell", ng.callbacks.ParametersLogger(filepath, append=False))
    optimizer.minimize(_func, verbosity=2)
    # pickling
    logger = callbacks.ParametersLogger(filepath)
    logs = logger.load_flattened()
    assert len(logs) == 2


def test_chaining_log_parameters(tmp_path: Path) -> None:
    filepath = tmp_path / "logs.txt"
    params = ng.p.Instrumentation(
        None, 2.0, blublu="blublu", array=ng.p.Array(shape=(3, 2)), multiobjective=False
    )
    zmethods = ["CauchyLHSSearch", "DE", "CMA"]
    ztmp1 = [ng.optimizers.registry[zmet] for zmet in zmethods]
    optmodel = ng.families.Chaining(ztmp1, [50, 50])  #
    optim = optmodel(parametrization=params, budget=100, num_workers=3)
    logger = ng.callbacks.ParametersLogger(filepath)
    optim.register_callback("tell", logger)
    optim.minimize(_func, verbosity=2)
    # read
    logger = callbacks.ParametersLogger(filepath)
    logs = logger.load_flattened()
    assert len(logs) == 100


def test_dump_callback(tmp_path: Path) -> None:
    filepath = tmp_path / "pickle.pkl"
    optimizer = optimizerlib.OnePlusOne(parametrization=2, budget=32)
    optimizer.register_callback("tell", ng.callbacks.OptimizerDump(filepath))
    cand = optimizer.ask()
    assert not filepath.exists()
    optimizer.tell(cand, 0)
    assert filepath.exists()


def test_progressbar_dump(tmp_path: Path) -> None:
    filepath = tmp_path / "pickle.pkl"
    optimizer = optimizerlib.OnePlusOne(parametrization=2, budget=32)
    optimizer.register_callback("tell", ng.callbacks.ProgressBar())
    for _ in range(8):
        cand = optimizer.ask()
        optimizer.tell(cand, 0)
    optimizer.dump(filepath)
    # should keep working after dump
    cand = optimizer.ask()
    optimizer.tell(cand, 0)
    # and be reloadable
    optimizer = optimizerlib.OnePlusOne.load(filepath)
    for _ in range(12):
        cand = optimizer.ask()
        optimizer.tell(cand, 0)


class _EarlyStoppingTestee:
    def __init__(self, val=None, multi=False) -> None:
        self.num_calls = 0
        self.val = val
        self.multi = False

    def __call__(self, *args, **kwds) -> tp.Union[float, tp.Tuple]:
        self.num_calls += 1
        if self.val is not None:
            return self.val if not self.multi else (self.val, self.val)
        return np.random.rand() if not self.multi else (np.random.rand(), np.random.rand())


def test_early_stopping() -> None:
    instrum = ng.p.Instrumentation(None, 2.0, blublu="blublu", array=ng.p.Array(shape=(3, 2)))
    func = _EarlyStoppingTestee()
    optimizer = optimizerlib.OnePlusOne(parametrization=instrum, budget=100)
    early_stopping = ng.callbacks.EarlyStopping(lambda opt: opt.num_ask > 3)
    optimizer.register_callback("ask", early_stopping)
    optimizer.register_callback("ask", ng.callbacks.EarlyStopping.timer(100))  # should not get triggered
    optimizer.minimize(func, verbosity=2)
    # num_ask is set at the end of ask, so the callback sees the old value.
    assert func.num_calls == 4
    # below functions are included in the docstring of EarlyStopping
    assert optimizer.current_bests["minimum"].mean < 12
    assert optimizer.recommend().loss < 12  # type: ignore

    # test for no improvement
    func = _EarlyStoppingTestee(5)
    optimizer = optimizerlib.OnePlusOne(parametrization=instrum, budget=100)
    no_imp_window = 7
    optimizer.register_callback(
        "ask", ng.callbacks.EarlyStopping.no_improvement_stopper(no_imp_window)
    )  # should get triggered
    optimizer.minimize(func, verbosity=2)
    assert func.num_calls == no_imp_window + 2

    # test for no improvement multi objective
    func = _EarlyStoppingTestee(5, multi=True)
    optimizer = optimizerlib.OnePlusOne(parametrization=instrum, budget=100)
    no_imp_window = 5
    optimizer.register_callback(
        "ask", ng.callbacks.EarlyStopping.no_improvement_stopper(no_imp_window)
    )  # should get triggered
    optimizer.minimize(func, verbosity=2)
    assert func.num_calls == no_imp_window + 2


def test_duration_criterion() -> None:
    optim = optimizerlib.OnePlusOne(2, budget=100)
    crit = ng.callbacks._DurationCriterion(0.01)
    assert not crit(optim)
    assert not crit(optim)
    assert not crit(optim)
    time.sleep(0.01)
    assert crit(optim)


def test_optimization_logger(caplog) -> None:
    instrum = ng.p.Instrumentation(
        None, 2.0, blublu="blublu", array=ng.p.Array(shape=(3, 2)), multiobjective=False
    )
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    logger = logging.getLogger(__name__)
    optimizer = optimizerlib.OnePlusOne(parametrization=instrum, budget=3)
    optimizer.register_callback(
        "tell",
        callbacks.OptimizationLogger(
            logger=logger, log_level=logging.INFO, log_interval_tells=10, log_interval_seconds=0.1
        ),
    )
    with caplog.at_level(logging.INFO):
        optimizer.minimize(_func, verbosity=2)
    assert (
        "After 0, recommendation is Instrumentation(Tuple(None,2.0),Dict(array=Array{(3,2)},blublu=blublu,multiobjective=False))"
        in caplog.text
    )


def test_optimization_logger_MOO(caplog) -> None:
    instrum = ng.p.Instrumentation(
        None, 2.0, blublu="blublu", array=ng.p.Array(shape=(3, 2)), multiobjective=True
    )
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    logger = logging.getLogger(__name__)
    optimizer = optimizerlib.OnePlusOne(parametrization=instrum, budget=3)
    optimizer.register_callback(
        "tell",
        callbacks.OptimizationLogger(
            logger=logger, log_level=logging.INFO, log_interval_tells=10, log_interval_seconds=0.1
        ),
    )
    with caplog.at_level(logging.INFO):
        optimizer.minimize(_func, verbosity=2)
    assert (
        "After 0, the respective minimum loss for each objective in the pareto front is [12. 12.]"
        in caplog.text
    )
