# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any
from pathlib import Path
import numpy as np
import nevergrad as ng
from . import optimizerlib
from . import callbacks


# pylint: disable=unused-argument
def _func(x: Any, y: Any, blublu: str, array: Any) -> float:
    return 12


def test_log_parameters(tmp_path: Path) -> None:
    filepath = tmp_path / "logs.txt"
    cases = [0, np.int(1), np.float(2.0), np.nan, float("inf"), np.inf]
    instrum = ng.p.Instrumentation(ng.p.Array(shape=(1,)),
                                   ng.p.Scalar(),
                                   blublu=ng.p.Choice(cases),
                                   array=ng.p.Array(shape=(3, 2)))
    optimizer = optimizerlib.NoisyOnePlusOne(parametrization=instrum, budget=32)
    optimizer.register_callback("tell", callbacks.ParametersLogger(filepath, append=False))
    optimizer.minimize(_func, verbosity=2)
    # pickling
    logger = callbacks.ParametersLogger(filepath)
    logs = logger.load_flattened()
    assert len(logs) == 32
    assert isinstance(logs[-1]["1"], float)
    assert len(logs[-1]) == 32
    logs = logger.load_flattened(max_list_elements=2)
    assert len(logs[-1]) == 24
    # deletion
    logger = callbacks.ParametersLogger(filepath, append=False)
    assert not logger.load()
