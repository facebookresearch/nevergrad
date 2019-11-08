# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import numpy as np
from . import variables
from .variables import wrap_arg


def test_softmax_categorical_deterministic() -> None:
    token = variables.SoftmaxCategorical(["blu", "blublu", "blublublu"], deterministic=True)
    assert token.data_to_arguments([1, 1, 1.01], deterministic=False) == wrap_arg("blublublu")


def test_softmax_categorical() -> None:
    np.random.seed(12)
    token = variables.SoftmaxCategorical(["blu", "blublu", "blublublu"])
    assert token.data_to_arguments([0.5, 1.0, 1.5]) == wrap_arg("blublu")
    assert token.data_to_arguments(token.arguments_to_data("blu"), deterministic=True) == wrap_arg("blu")


def test_ordered_discrete() -> None:
    token = variables.OrderedDiscrete(["blu", "blublu", "blublublu"])
    assert token.data_to_arguments([5]) == wrap_arg("blublublu")
    assert token.data_to_arguments([0]) == wrap_arg("blublu")
    assert token.data_to_arguments(token.arguments_to_data("blu"), deterministic=True) == wrap_arg("blu")


def test_gaussian() -> None:
    token = variables.Gaussian(1, 3)
    assert token.data_to_arguments([.5]) == wrap_arg(2.5)
    data = token.arguments_to_data(12)
    print(data)
    assert token.data_to_arguments(data) == wrap_arg(12)


def test_scalar() -> None:
    token = variables.Scalar(int)
    assert token.data_to_arguments([.7]) == wrap_arg(1)
    assert token.arguments_to_data(1).tolist() == [1.]


def test_array_as_ascalar() -> None:
    var = variables.Array(1).exponentiated(10, -1).asscalar()
    data = np.array([2])
    output = var.data_to_arguments(data)
    assert output == wrap_arg(0.01)
    np.testing.assert_almost_equal(var.arguments_to_data(*output[0], **output[1]), data)
    #  int
    var = variables.Array(1).asscalar(int)
    assert var.data_to_arguments(np.array([.4])) == wrap_arg(0)
    assert var.data_to_arguments(np.array([-.4])) == wrap_arg(0)
    output = var.data_to_arguments(np.array([.6]))
    assert output == wrap_arg(1)
    assert type(output[0][0]) == int  # pylint: disable=unidiomatic-typecheck
    # errors
    with pytest.raises(RuntimeError):
        variables.Array(1).asscalar(int).asscalar(float)
    with pytest.raises(RuntimeError):
        variables.Array(2).asscalar(int)
    with pytest.raises(ValueError):
        variables.Array(1).asscalar(np.int64)  # type: ignore


def test_array() -> None:
    var = variables.Array(2, 2).affined(1000000).bounded(3, 5, transform="arctan")
    data = np.array([-10, 10, 0, 0])
    output = var.data_to_arguments(data)
    np.testing.assert_almost_equal(output[0][0], [[3., 5], [4, 4]])
    np.testing.assert_almost_equal(var.arguments_to_data(*output[0], **output[1]), data)


@pytest.mark.parametrize("value,expected", [(0, 0.01), (10, 0.1), (-10, 0.001), (20, 0.1), (9, 0.07943)])  # type: ignore
def test_log(value: float, expected: float) -> None:
    var = variables.Log(0.001, 0.1)
    out = var.data_to_arguments(np.array([value]))
    np.testing.assert_approx_equal(out[0][0], expected, significant=4)
    repr(var)


def test_log_int() -> None:
    var = variables.Log(300, 10000, dtype=int)
    out = var.data_to_arguments(np.array([0]))
    assert out[0][0] == 1732


# note: 0.9/0.9482=0.9482/0.999
@pytest.mark.parametrize("value,expected", [(0, 0.9482), (-11, 0.9), (10, 0.999)])  # type: ignore
def test_log_9(value: float, expected: float) -> None:
    var = variables.Log(0.9, 0.999)
    out = var.data_to_arguments(np.array([value]))
    np.testing.assert_approx_equal(out[0][0], expected, significant=4)
