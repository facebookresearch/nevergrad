# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import pytest
import numpy as np
import nevergrad as ng
from . import constraints


def function(*args: float) -> tp.Any:
    if len(args) == 1:
        return args[0]
    return args


@pytest.mark.parametrize("num", (1, 3))  # type: ignore
def test_constraint(num: int) -> None:
    parameter = ng.p.Instrumentation(*(ng.p.Scalar(np.random.randn()) for _ in range(num)))
    constrained = constraints.Constraint(function)(parameter)
    # check basic layer functionalities
    layer: constraints.Constraint = constrained._layers[-1]  # type: ignore
    assert layer.function(*([1] * num)) == 1 if num == 1 else [1] * num
    assert layer.function(*([-1] * num)) == 0 if num == 1 else [0] * num
    assert (
        np.mean([x < 0.1 for x in constrained.args]) > 0.5
    ), constrained.args  # some slack to avoid flakiness
