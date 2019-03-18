# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Tuple, Dict
import numpy as np
from ..common import testing
from . import variables
from . import core


def test_instrumentation() -> None:
    instru = core.Instrumentation(variables.Gaussian(0, 1),
                                  3,
                                  b=variables.SoftmaxCategorical([0, 1, 2, 3]),
                                  a=variables.OrderedDiscrete([0, 1, 2, 3]))
    np.testing.assert_equal(instru.dimension, 6)
    data = instru.arguments_to_data(4, 3, a=0, b=3)
    np.testing.assert_array_almost_equal(data, [4, -1.1503, 0, 0, 0, .5878], decimal=4)
    args, kwargs = instru.data_to_arguments(data, deterministic=True)
    testing.printed_assert_equal((args, kwargs), ((4., 3), {'a': 0, 'b': 3}))
    assert ", 3, a=Ordered" in repr(instru), f"Erroneous representation {instru}"
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
    instru2 = core.Instrumentation(*instru.args, **instru.kwargs)
    data = np.random.normal(0, 1, size=6)
    testing.printed_assert_equal(instru2.data_to_arguments(data, deterministic=True),
                                 instru.data_to_arguments(data, deterministic=True))
    # check naming
    testing.printed_assert_equal("G(0,1),3,a=OD(0,1,2,3),b=SC(0,1,2,3|0)", instru.name)
    testing.printed_assert_equal("blublu", instru.with_name("blublu").name)


def test_instrumentation_init_error() -> None:
    variable = variables.Gaussian(0, 1)
    np.testing.assert_raises(AssertionError, core.Instrumentation, variable, variable)


def _arg_return(*args: Any, **kwargs: Any) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    return args, kwargs


def test_instrumented_function() -> None:
    ifunc = core.InstrumentedFunction(_arg_return, variables.SoftmaxCategorical([1, 12]), "constant",
                                      variables.Gaussian(0, 1, [2, 2]), constkwarg="blublu",
                                      plop=variables.SoftmaxCategorical([3, 4]))
    np.testing.assert_equal(ifunc.dimension, 8)
    data = [-100., 100, 1, 2, 3, 4, 100, -100]
    args, kwargs = ifunc(np.array(data))
    testing.printed_assert_equal(args, [12, "constant", [[1, 2], [3, 4]]])
    testing.printed_assert_equal(kwargs, {"constkwarg": "blublu", "plop": 3})
    testing.printed_assert_equal(ifunc.descriptors, {
        "dimension": 8, "name": "_arg_return", "function_class": "InstrumentedFunction",
        "instrumentation": "SC(1,12|0),constant,G(0,1),constkwarg=blublu,plop=SC(3,4|0)"})
    print(ifunc.get_summary(data))


def test_instrumented_function_kwarg_order() -> None:
    ifunc = core.InstrumentedFunction(_arg_return, kw4=variables.SoftmaxCategorical([1, 0]), kw2="constant",
                                      kw3=variables.Array(2, 2), kw1=variables.Gaussian(2, 2))
    np.testing.assert_equal(ifunc.dimension, 7)
    data = np.array([-1, 1, 2, 3, 4, 100, -100])
    _, kwargs = ifunc(data)
    testing.printed_assert_equal(kwargs, {"kw1": 0, "kw2": "constant", "kw3": [[1, 2], [3, 4]], "kw4": 1})


class _Callable:

    def __call__(self, x: float, y: float = 0) -> float:
        return abs(x + y)


def test_callable_instrumentation() -> None:
    ifunc = core.InstrumentedFunction(lambda x: x**2, variables.Gaussian(2, 2))
    np.testing.assert_equal(ifunc.descriptors["name"], "<lambda>")
    ifunc = core.InstrumentedFunction(_Callable(), variables.Gaussian(2, 2))
    np.testing.assert_equal(ifunc.descriptors["name"], "_Callable")


def test_deterministic_data_to_arguments() -> None:
    ifunc = core.InstrumentedFunction(_Callable(), variables.SoftmaxCategorical([0, 1, 2, 3]),
                                      y=variables.SoftmaxCategorical([0, 1, 2, 3]))
    data = [.01, 0, 0, 0, .01, 0, 0, 0]
    for _ in range(20):
        args, kwargs = ifunc.data_to_arguments(data, deterministic=True)
        testing.printed_assert_equal(args, [0])
        testing.printed_assert_equal(kwargs, {"y": 0})
    arg_sum, kwarg_sum = 0, 0
    for _ in range(24):
        args, kwargs = ifunc.data_to_arguments(data, deterministic=False)
        arg_sum += args[0]
        kwarg_sum += kwargs["y"]
    assert arg_sum != 0
    assert kwarg_sum != 0
