# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List
import numpy as np
from ..common import testing
from . import variables as var
from . import multivariables as mvar
from .core import Variable


@testing.parametrized(
    empty=([], [], [])
)
def test_split_data(tokens: List[Variable], data: List[float], expected: List[List[float]]) -> None:
    instru = mvar.Instrumentation(*tokens)
    output = instru._split_data(np.array(data))
    testing.printed_assert_equal(output, expected)


def test_nested_variables_data_to_arguments() -> None:
    instru = mvar.NestedVariables(var.SoftmaxCategorical(list(range(5))), var.Gaussian(3, 4))
    values = instru.data_to_arguments([0, 200, 0, 0, 0, 2])
    expected: Any = (tuple(var.wrap_arg(x) for x in (1, 11)), {})
    assert values == expected


def test_instrumentation_data_to_arguments() -> None:
    tokens = [var.SoftmaxCategorical(list(range(5))), var.Gaussian(3, 4)]
    instru = mvar.Instrumentation(*tokens)
    values = instru.data_to_arguments([0, 200, 0, 0, 0, 2])[0]
    assert values == (1, 11)
    np.testing.assert_raises(ValueError, instru.data_to_arguments, tokens, [0, 200, 0, 0, 0, 2, 3])


def test_instrumentation() -> None:
    instru = mvar.Instrumentation(var.Gaussian(0, 1), 3, b=var.SoftmaxCategorical([0, 1, 2, 3]), a=var.OrderedDiscrete([0, 1, 2, 3]))
    np.testing.assert_equal(instru.dimension, 6)
    instru2 = mvar.Instrumentation(var.Gaussian(0, 1), 3, b=var.SoftmaxCategorical([0, 1, 2, 3]), a=var.UnorderedDiscrete([0, 1, 2, 3]))
    np.testing.assert_equal(instru2.dimension, 6)
    data = instru.arguments_to_data(4, 3, a=0, b=3)
    np.testing.assert_array_almost_equal(data, [4, -1.1503, 0, 0, 0, 0.5878], decimal=4)
    args, kwargs = instru.data_to_arguments(data, deterministic=True)
    testing.printed_assert_equal((args, kwargs), ((4.0, 3), {"a": 0, "b": 3}))
    assert ", 3, a=Ordered" in repr(instru), f"Erroneous representation {instru}"
    # check deterministic
    data = np.array([0.0, 0, 0, 0, 0, 0])
    total = 0
    for _ in range(24):
        total += instru.data_to_arguments(data, deterministic=True)[1]["b"]
    np.testing.assert_equal(total, 0)
    # check stochastic
    for _ in range(24):
        total += instru.data_to_arguments(data, deterministic=False)[1]["b"]
    assert total != 0
    # check duplicate
    instru2 = mvar.Instrumentation(*instru.args, **instru.kwargs)
    data = np.random.normal(0, 1, size=6)
    testing.printed_assert_equal(instru2.data_to_arguments(data, deterministic=True), instru.data_to_arguments(data, deterministic=True))
    # check naming
    testing.printed_assert_equal("G(0,1),3,a=OD(0,1,2,3),b=SC(0,1,2,3|0)", instru.name)
    testing.printed_assert_equal("blublu", instru.with_name("blublu").name)


def test_instrumentation_copy() -> None:
    instru = mvar.Instrumentation(var.Gaussian(0, 1), 3, b=var.SoftmaxCategorical(list(range(1000)))).with_name("bidule")
    instru.set_cheap_constraint_checker(lambda *args, **kwargs: False)
    copied = instru.copy()
    assert copied.name == "bidule"
    assert copied.random_state is not instru.random_state
    assert str(copied.variables) == str(instru.variables)
    # test that variables do not hold a random state / interfere
    instru.random_state = np.random.RandomState(12)
    copied.random_state = np.random.RandomState(12)
    kwargs1 = instru.data_to_arguments([0] * 1001)[1]
    kwargs2 = copied.data_to_arguments([0] * 1001)[1]
    assert kwargs1 == kwargs2
    assert not copied.cheap_constraint_check(12)


def test_instrumentation_split() -> None:
    instru = mvar.Instrumentation(var.Gaussian(0, 1), 3, b=var.SoftmaxCategorical([0, 1, 2, 3]), a=var.OrderedDiscrete([0, 1, 2, 3]))
    splitted = instru._split_data(np.array([0, 1, 2, 3, 4, 5]))
    np.testing.assert_equal([x.tolist() for x in splitted], [[0], [], [1], [2, 3, 4, 5]])  # order of kwargs is alphabetical


def test_instrumentation_init_error() -> None:
    variable = var.Gaussian(0, 1)
    np.testing.assert_raises(AssertionError, mvar.Instrumentation, variable, variable)
