# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from . import variables


def test_softmax_categorical_deterministic() -> None:
    token = variables.SoftmaxCategorical(["blu", "blublu", "blublublu"], deterministic=True)
    np.testing.assert_equal(token.data_to_argument([1, 1, 1.01], deterministic=False), "blublublu")


def test_softmax_categorical() -> None:
    np.random.seed(12)
    token = variables.SoftmaxCategorical(["blu", "blublu", "blublublu"])
    np.testing.assert_equal(token.data_to_argument([.5, 1, 2.]), "blublu")
    np.testing.assert_equal(token.data_to_argument(token.argument_to_data("blu"), deterministic=True), "blu")


def test_ordered_discrete() -> None:
    token = variables.OrderedDiscrete(["blu", "blublu", "blublublu"])
    np.testing.assert_equal(token.data_to_argument([5]), "blublublu")
    np.testing.assert_equal(token.data_to_argument([0]), "blublu")
    np.testing.assert_equal(token.data_to_argument(token.argument_to_data("blu"), deterministic=True), "blu")


def test_gaussian() -> None:
    token = variables.Gaussian(1, 3)
    np.testing.assert_equal(token.data_to_argument([.5]), 2.5)
    np.testing.assert_equal(token.data_to_argument(token.argument_to_data(12)), 12)


def test_array_float() -> None:
    var = variables.Array(1).exponentiated(10, -1).asfloat()
    data = np.array([2])
    output = var.data_to_argument(data)
    np.testing.assert_equal(output, 0.01)
    np.testing.assert_almost_equal(var.argument_to_data(output), data)


def test_array() -> None:
    var = variables.Array(2, 2).affined(1000000).bounded(3, 5, transform="arctan")
    data = np.array([-10, 10, 0, 0])
    output = var.data_to_argument(data)
    np.testing.assert_almost_equal(output, [[3., 5], [4, 4]])
    np.testing.assert_almost_equal(var.argument_to_data(output), data)
