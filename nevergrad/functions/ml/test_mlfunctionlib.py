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
    np.testing.assert_almost_equal(y1, 0.00118061025851494)
    np.testing.assert_almost_equal(y3, 0.0009174743914262424)
    np.testing.assert_array_almost_equal(y3, y4)  # should be equal

    # Testing a decision tree.
    np.random.seed(17)
    func2 = mlfunctionlib.MLTuning("decision_tree", 2)
    np.testing.assert_almost_equal(func2(depth=3, criterion="mse", min_samples_split=0.001),
                                   0.00468323299294414)

    # Testing a multi-layer perceptron.
    np.random.seed(17)
    func3 = mlfunctionlib.MLTuning("mlp", 2)
    np.testing.assert_almost_equal(
        func3(activation="relu", solver="adam", alpha=0.01, learning_rate="constant"),
        0.0038580300812402378)

    # Testing a classifier choosing between a multi-layer perceptron and a decision tree.
    np.random.seed(17)
    func4 = mlfunctionlib.MLTuning("any", 2)
    np.testing.assert_almost_equal(
        func4(activation="relu", solver="adam", alpha=0.01, learning_rate="constant",
              depth=3, criterion="mse", min_samples_split=0.001, regressor="mlp"),
        0.0038580300812402378)
    np.testing.assert_almost_equal(
        func4(activation="relu", solver="adam", alpha=0.01, learning_rate="constant",
              depth=3, criterion="mse", min_samples_split=0.001,
              regressor="decision_tree"), 0.00468323299294414)

    # Testing a decision tree on SKLearn's Boston.
    np.random.seed(17)
    func5 = mlfunctionlib.MLTuning("decision_tree", data_dimension=None, dataset_name="boston")
    np.testing.assert_almost_equal(func5(depth=5, criterion="mse", min_samples_split=0.001),
                                   36.94674000207121)

    # Testing a decision tree on SKLearn's Diabetes.
    np.random.seed(17)
    func6 = mlfunctionlib.MLTuning("decision_tree", data_dimension=None, dataset_name="diabetes")
    np.testing.assert_almost_equal(func6(depth=5, criterion="mse", min_samples_split=0.001),
                                   5449.271797229855)


