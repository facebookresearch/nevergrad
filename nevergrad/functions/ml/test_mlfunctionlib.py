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
    func = mlfunctionlib.MLTuning("decision_tree_depth_regression", 1)
    x: int = 3
    y1 = func(x)  # returns a float
    y2 = func(x)  # returns the same float
    np.testing.assert_array_almost_equal(y1, y2)
    y3 = func.evaluation_function(x)   # returns a float
    y4 = func.evaluation_function(x)   # returns the same float
    np.testing.assert_array_almost_equal(y3, y4)  # should be equal
    func2 = mlfunctionlib.MLTuning("decision_tree_regression, 2")
    func2(depth=3, criterion="mse", min_samples_split=0.001)
    func3 = mlfunctionlib.MLTuning("nn", 2)
    func3(activation="relu", solver="adam", alpha=0.01, learning_rate="constant")
