# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any
from pathlib import Path
import nevergrad as ng
from . import optimizerlib
from . import callbacks


# pylint: disable=unused-argument
def _func(x: Any, y: Any, blublu: str, array: Any) -> float:
    return float(blublu == "a")


def test_log_parameters(tmp_path: Path) -> None:
    filepath = tmp_path / "logs.txt"
    instrum = ng.Instrumentation(ng.var.Array(1),
                                 ng.var.Scalar(),
                                 blublu=ng.var.SoftmaxCategorical(["a", "b"]),
                                 array=ng.var.Array(3, 2))
    optimizer = optimizerlib.OnePlusOne(instrumentation=instrum, budget=10, num_workers=5)
    optimizer.register_callback("tell", callbacks.ParametersLogger(filepath))
    optimizer.minimize(_func, verbosity=2)
    # pickling
    logger = callbacks.ParametersLogger(filepath)
    logs = logger.load()
    assert len(logs) == 10
