# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
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
    cases = [0, np.int(1), np.float(2.0), np.nan, float("inf"), np.inf]
    instrum = ng.p.Instrumentation(
        ng.p.Array(shape=(1,)).set_mutation(custom=ng.p.mutation.Translation()),
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
    assert len(logs[-1]) == 30
    logs = logger.load_flattened(max_list_elements=2)
    assert len(logs[-1]) == 26
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


def test_early_stopping() -> None:
    instrum = ng.p.Instrumentation(
        None, 2.0, blublu="blublu", array=ng.p.Array(shape=(3, 2)), multiobjective=True
    )
    optimizer = optimizerlib.OnePlusOne(parametrization=instrum, budget=100)
    early_stopping = ng.callbacks.EarlyStopping(lambda opt: opt.num_ask > 3)
    optimizer.register_callback("ask", early_stopping)
    optimizer.minimize(_func, verbosity=2)
    # below functions are inlcuded in the docstring
    assert optimizer.current_bests["minimum"].mean < 12
    assert optimizer.recommend().loss < 12  # type: ignore
