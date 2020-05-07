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
    func = mlfunctionlib.MLTuning("decision_tree_depth", 1)
    x: int = 3
    y1 = func(x)  # returns a float
    y2 = func(x)  # returns the same float
    np.testing.assert_array_almost_equal(y1, y2)
    y3 = func.evaluation_function(x)   # returns a float
    y4 = func.evaluation_function(x)   # returns the same float
    np.testing.assert_almost_equal(y1, 0.00118061025851494)
    np.testing.assert_almost_equal(y3, 4.6543281260653915)
    np.testing.assert_array_almost_equal(y3, y4)  # should be equal
    func2 = mlfunctionlib.MLTuning("decision_tree", 2)
    np.testing.assert_almost_equal(func2(depth=3, criterion="mse", min_samples_split=0.001),
                                   0.011687671501421443)
    func3 = mlfunctionlib.MLTuning("mlp", 2)
    np.testing.assert_almost_equal(
        func3(activation="relu", solver="adam", alpha=0.01, learning_rate="constant"),
        0.005295441439915157)
    func4 = mlfunctionlib.MLTuning("any", 2)
    np.testing.assert_almost_equal(
        func4(activation="relu", solver="adam", alpha=0.01, learning_rate="constant",
              depth=3, criterion="mse", min_samples_split=0.001, regressor="mlp"),
        0.005295441439915157)
    np.testing.assert_almost_equal(
        func4(activation="relu", solver="adam", alpha=0.01, learning_rate="constant",
              depth=3, criterion="mse", min_samples_split=0.001,
              regressor="decision_tree"), 0.011687671501421444)

