# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict
import numpy as np
import pytest
from nevergrad.common import testing
from nevergrad.parametrization import parameter as p
from . import mlfunctionlib


def test_ml_tuning() -> None:
    func = mlfunctionlib.MLTuning("1d_decision_tree_regression")
    x: int = 3
    y1 = func(x)  # returns a float
    y2 = func(x)  # returns the same float
    np.testing.assert_array_almost_equal(y1, y2)
    y3 = func.evaluation_function(x)   # returns a float
    y4 = func.evaluation_function(x)   # returns the same float
    np.testing.assert_array_almost_equal(y3, y4)  # should be equal
    func2 = mlfunctionlib.MLTuning("3d_decision_tree_regression_full")
    func2(depth=3, criterion="mse", min_samples_split=0.001)
