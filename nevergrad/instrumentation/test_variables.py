# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Tuple, Dict
import numpy as np
from ..common import testing
from . import variables


def test_soft_discrete() -> None:
    np.random.seed(12)
    token = variables.SoftmaxCategorical(["blu", "blublu", "blublublu"])
    np.testing.assert_equal(token.process([.5, 1, 2.]), "blublu")
    np.testing.assert_equal(token.process(token.process_arg("blu"), deterministic=True), "blu")


def test_hard_discrete() -> None:
    token = variables.OrderedDiscrete(["blu", "blublu", "blublublu"])
    np.testing.assert_equal(token.process([5]), "blublublu")
    np.testing.assert_equal(token.process([0]), "blublu")
    np.testing.assert_equal(token.process(token.process_arg("blu"), deterministic=True), "blu")


def test_gaussian() -> None:
    token = variables.Gaussian(1, 3)
    np.testing.assert_equal(token.process([.5]), 2.5)
    np.testing.assert_equal(token.process(token.process_arg(12)), 12)


def test_instrumentation() -> None:
    instru = variables.Instrumentation(variables.Gaussian(0, 1),
                                       3,
                                       b=variables.SoftmaxCategorical([0, 1, 2, 3]),
                                       a=variables.OrderedDiscrete([0, 1, 2, 3]))
    np.testing.assert_equal(instru.dimension, 6)
    data = instru.arguments_to_data(4, 3, a=0, b=3)
    np.testing.assert_array_almost_equal(data, [4, -1.1503, 0, 0, 0, .5878], decimal=4)
    args, kwargs = instru.data_to_arguments(data, deterministic=True)
    testing.printed_assert_equal((args, kwargs), ((4., 3), {'a': 0, 'b': 3}))
    # check deterministic
    data = [0, 0, 0, 0, 0, 0]
    total = 0
    for _ in range(24):
        total += instru.data_to_arguments(data, deterministic=True)[1]["b"]
    np.testing.assert_equal(total, 0)
    # check stochastic
    for _ in range(24):
        total += instru.data_to_arguments(data, deterministic=False)[1]["b"]
    assert total != 0
    # check duplicate
    instru2 = variables.Instrumentation(*instru.args, **instru.kwargs)
    data = np.random.normal(0, 1, size=6)
    testing.printed_assert_equal(instru2.data_to_arguments(data, deterministic=True),
                                 instru.data_to_arguments(data, deterministic=True))


def _arg_return(*args: Any, **kwargs: Any) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    return args, kwargs


def test_instrumented_function() -> None:
    ifunc = variables.InstrumentedFunction(_arg_return, variables.SoftmaxCategorical([1, 12]), "constant",
                                           variables.Gaussian(0, 1, [2, 2]), constkwarg="blublu",
                                           plop=variables.SoftmaxCategorical([3, 4]))
    np.testing.assert_equal(ifunc.dimension, 8)
    data = [-100, 100, 1, 2, 3, 4, 100, -100]
    args, kwargs = ifunc(data)
    testing.printed_assert_equal(args, [12, "constant", [[1, 2], [3, 4]]])
    testing.printed_assert_equal(kwargs, {"constkwarg": "blublu", "plop": 3})
    testing.printed_assert_equal(ifunc.descriptors, {"dimension": 8, "name": "_arg_return", "instrumented": "arg0,arg2,plop",
                                                     "function_class": "InstrumentedFunction", "transform": None})
    print(ifunc.get_summary(data))


def test_instrumented_function_kwarg_order() -> None:
    ifunc = variables.InstrumentedFunction(_arg_return, kw4=variables.SoftmaxCategorical([1, 0]), kw2="constant",
                                           kw3=variables.Gaussian(0, 1, [2, 2]), kw1=variables.Gaussian(2, 2))
    np.testing.assert_equal(ifunc.dimension, 7)
    data = [-1, 1, 2, 3, 4, 100, -100]
    _, kwargs = ifunc(data)
    testing.printed_assert_equal(kwargs, {"kw1": 0, "kw2": "constant", "kw3": [[1, 2], [3, 4]], "kw4": 1})


class _Callable:

    def __call__(self, x: float, y: float = 0) -> float:
        return abs(x + y)


def test_callable_instrumentation() -> None:
    ifunc = variables.InstrumentedFunction(lambda x: x**2, variables.Gaussian(2, 2))
    np.testing.assert_equal(ifunc.descriptors["name"], "<lambda>")
    ifunc = variables.InstrumentedFunction(_Callable(), variables.Gaussian(2, 2))
    np.testing.assert_equal(ifunc.descriptors["name"], "_Callable")


def test_deterministic_convert_to_args() -> None:
    ifunc = variables.InstrumentedFunction(_Callable(), variables.SoftmaxCategorical([0, 1, 2, 3]),
                                           y=variables.SoftmaxCategorical([0, 1, 2, 3]))
    data = [.01, 0, 0, 0, .01, 0, 0, 0]
    for _ in range(20):
        args, kwargs = ifunc.convert_to_arguments(data, deterministic=True)
        testing.printed_assert_equal(args, [0])
        testing.printed_assert_equal(kwargs, {"y": 0})
    arg_sum, kwarg_sum = 0, 0
    for _ in range(24):
        args, kwargs = ifunc.convert_to_arguments(data, deterministic=False)
        arg_sum += args[0]
        kwarg_sum += kwargs["y"]
    assert arg_sum != 0
    assert kwarg_sum != 0
