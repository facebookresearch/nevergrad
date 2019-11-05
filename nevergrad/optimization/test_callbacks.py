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
    instrum = ng.Instrumentation(ng.var.Array(1),
                                 ng.var.Scalar(),
                                 blublu=ng.var.SoftmaxCategorical(cases),
                                 array=ng.var.Array(3, 2))
    optimizer = optimizerlib.OnePlusOne(instrumentation=instrum, budget=32, num_workers=5)
    optimizer.register_callback("tell", callbacks.ParametersLogger(filepath))
    optimizer.minimize(_func, verbosity=2)
    # pickling
    logger = callbacks.ParametersLogger(filepath)
    logs = logger.load()
    assert len(logs) == 32
    assert isinstance(logs[-1]["#arg1"], float)
    assert len(logs[-1]) == 15
    logs = logger.load(max_list_elements=3)
    assert len(logs[-1]) == 12
