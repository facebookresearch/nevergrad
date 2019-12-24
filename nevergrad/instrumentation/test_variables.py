# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import pytest
import numpy as np
from nevergrad.parametrization import parameter as p
from nevergrad.parametrization.test_parameter import check_parameter_features
from . import variables
from .variables import wrap_arg


def test_softmax_categorical_deterministic() -> None:
    token = p.Instrumentation(variables.SoftmaxCategorical(["blu", "blublu", "blublublu"], deterministic=True))
    assert token.data_to_arguments([1, 1, 1.01], deterministic=False) == wrap_arg("blublublu")


def test_softmax_categorical() -> None:
    np.random.seed(12)
    token = p.Instrumentation(variables.SoftmaxCategorical(["blu", "blublu", "blublublu"]))
    assert token.data_to_arguments([0.5, 1.0, 1.5]) == wrap_arg("blublu")
    assert token.data_to_arguments(token.arguments_to_data("blu"), deterministic=True) == wrap_arg("blu")


def test_ordered_discrete() -> None:
    token = p.Instrumentation(variables.OrderedDiscrete(["blu", "blublu", "blublublu"]))
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


@pytest.mark.parametrize(  # type: ignore
    "var,data,expected",
    [
        (variables.Log(0.9, 0.999), [0], 0.9482),
        (variables.Array(2).affined(10, 100), [0, 3], [100, 130]),
        (variables.Scalar().affined(10, 100).bounded(-200, 200), [0], 198.7269),
        (variables.Scalar(int).affined(10, 100).bounded(-200, 200), [0], 199),
        (variables.Scalar().exponentiated(10, -1), [1], 0.1),
        (variables.Scalar().exponentiated(2, 3), [4], 4096),
        (variables.Scalar().affined(10, 100).bounded(-200, 200), [-10], 0),
        (variables.Scalar().affined(10, 100).bounded(-200, 200, transform="clipping"), [1], 110),
        (variables.Gaussian(3, 5, shape=(2,)), [-2, 1], [-7, 8]),
        (variables.Gaussian(3, 5), [-2], -7),
        (p.Instrumentation(variables.OrderedDiscrete(list(range(100)))), [1.4], 91),
    ]
)
def test_expected_value(var: variables.Variable, data: tp.List[float], expected: tp.Any) -> None:
    check_parameter_features(var)
    out = var.data_to_arguments(np.array(data))[0][0]
    if isinstance(out, np.ndarray):
        np.testing.assert_array_almost_equal(out, expected)
    else:
        np.testing.assert_approx_equal(out, expected, significant=4)
