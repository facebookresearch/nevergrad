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
    # Testing a decision tree with only depth as a parameter.
    np.random.seed(17)
    func = mlfunctionlib.MLTuning("decision_tree_depth", 1)
    x: int = 3
    y1 = func(x)  # returns a float
    y2 = func(x)  # returns the same float
    np.testing.assert_array_almost_equal(y1, y2)
    y3 = func.evaluation_function(x)   # returns a float
    y4 = func.evaluation_function(x)   # returns the same float
    np.testing.assert_almost_equal(y1, 0.0011839077714085452)
    np.testing.assert_almost_equal(y3, 0.0009174743914262424)
    np.testing.assert_array_almost_equal(y3, y4)  # should be equal

    # Testing a decision tree.
    np.random.seed(17)
    func2 = mlfunctionlib.MLTuning("decision_tree", 2)
    np.testing.assert_almost_equal(func2(depth=3, criterion="mse", min_samples_split=0.001),
                                   0.004715789566064921)

    # Testing a multi-layer perceptron.
    func.rng.seed(17)
    func3 = mlfunctionlib.MLTuning("mlp", 2)
    np.testing.assert_almost_equal(
        func3(activation="relu", solver="adam", alpha=0.01, learning_rate="constant"),
        0.003822067429702949)

    # Testing a classifier choosing between a multi-layer perceptron and a decision tree.
    func.rng.seed(17)
    func4 = mlfunctionlib.MLTuning("any", 2)
    np.testing.assert_almost_equal(
        func4(activation="relu", solver="adam", alpha=0.01, learning_rate="constant",
              depth=3, criterion="mse", min_samples_split=0.001, regressor="mlp"),
        0.0038987748201582714)
    np.testing.assert_almost_equal(
        func4(activation="relu", solver="adam", alpha=0.01, learning_rate="constant",
              depth=3, criterion="mse", min_samples_split=0.001,
              regressor="decision_tree"), 0.004955893558348708)

    # Testing a decision tree on SKLearn's Boston.
    func.rng.seed(17)
    func5 = mlfunctionlib.MLTuning("decision_tree", data_dimension=None, dataset="boston")
    np.testing.assert_almost_equal(func5(depth=5, criterion="mse", min_samples_split=0.001),
                                   34.460213262464116)

    # Testing a decision tree on SKLearn's Diabetes.
    func.rng.seed(17)
    func6 = mlfunctionlib.MLTuning("decision_tree", data_dimension=None, dataset="diabetes")
    np.testing.assert_almost_equal(func6(depth=5, criterion="mse", min_samples_split=0.001),
                                   5169.578358315828)


    # Testing a decision tree on cosinus.
    func.rng.seed(17)
    func7 = mlfunctionlib.MLTuning("decision_tree", data_dimension=3, dataset="artificialcos")
    np.testing.assert_almost_equal(func7(depth=5, criterion="mse", min_samples_split=0.001),
                                   0.0004757830416897488)


