# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import numpy as np
from nevergrad.instrumentation import Instrumentation
from nevergrad.instrumentation import variables as var
from nevergrad.common import testing
from . import base


def _arg_return(*args: tp.Any, **kwargs: tp.Any) -> tp.Tuple[tp.Tuple[tp.Any, ...], tp.Dict[str, tp.Any]]:
    return args, kwargs


def test_experimented_function() -> None:
    ifunc = base.ExperimentFunction(_arg_return, Instrumentation(  # type: ignore
        var.SoftmaxCategorical([1, 12]),
        "constant",
        var.Gaussian(0, 1, [2, 2]),
        constkwarg="blublu",
        plop=var.SoftmaxCategorical([3, 4]),
    ))
    np.testing.assert_equal(ifunc.dimension, 8)
    data = [-100.0, 100, 1, 2, 3, 4, 100, -100]
    args0, kwargs0 = ifunc.parametrization.data_to_arguments(np.array(data))
    output = ifunc(*args0, **kwargs0)  # this is very stupid and should be removed when Parameter is in use
    args: tp.Any = output[0]  # type: ignore
    kwargs: tp.Any = output[1]  # type: ignore
    testing.printed_assert_equal(args, [12, "constant", [[1, 2], [3, 4]]])
    testing.printed_assert_equal(kwargs, {"constkwarg": "blublu", "plop": 3})
    instru_str = ("Instrumentation(Tuple(SoftmaxCategorical(choices=Tuple(1,12),"
                  "weights=Array{(2,)}[recombination=average,sigma=1.0]),constant,G(0,1)),"
                  "Dict(constkwarg=blublu,plop=SoftmaxCategorical(choices=Tuple(3,4),"
                  "weights=Array{(2,)}[recombination=average,sigma=1.0])))")
    testing.printed_assert_equal(
        ifunc.descriptors,
        {
            "dimension": 8,
            "name": "_arg_return",
            "function_class": "ExperimentFunction",
            "instrumentation": instru_str,
        },
    )


def test_instrumented_function_kwarg_order() -> None:
    ifunc = base.ExperimentFunction(_arg_return, Instrumentation(  # type: ignore
        kw4=var.SoftmaxCategorical([1, 0]), kw2="constant", kw3=var.Array(2, 2), kw1=var.Gaussian(2, 2)
    ))
    np.testing.assert_equal(ifunc.dimension, 7)
    data = np.array([-1, 1, 2, 3, 4, 100, -100])
    args0, kwargs0 = ifunc.parametrization.data_to_arguments(data)
    # this is very stupid and should be removed when Parameter is in use
    kwargs: tp.Any = ifunc(*args0, **kwargs0)[1]   # type: ignore
    testing.printed_assert_equal(kwargs, {"kw1": 0, "kw2": "constant", "kw3": [[1, 2], [3, 4]], "kw4": 1})


class _Callable:
    def __call__(self, x: float, y: float = 0) -> float:
        return abs(x + y)


def test_callable_instrumentation() -> None:
    ifunc = base.ExperimentFunction(lambda x: x ** 2, var.Gaussian(2, 2))  # type: ignore
    np.testing.assert_equal(ifunc.descriptors["name"], "<lambda>")
    ifunc = base.ExperimentFunction(_Callable(), var.Gaussian(2, 2))
    np.testing.assert_equal(ifunc.descriptors["name"], "_Callable")


def test_deterministic_data_to_arguments() -> None:
    instru = Instrumentation(var.SoftmaxCategorical([0, 1, 2, 3]), y=var.SoftmaxCategorical([0, 1, 2, 3]))
    ifunc = base.ExperimentFunction(_Callable(), instru)
    data = [0.01, 0, 0, 0, 0.01, 0, 0, 0]
    for _ in range(20):
        args, kwargs = ifunc.parametrization.data_to_arguments(data, deterministic=True)
        testing.printed_assert_equal(args, [0])
        testing.printed_assert_equal(kwargs, {"y": 0})
    arg_sum, kwarg_sum = 0, 0
    for _ in range(24):
        args, kwargs = ifunc.parametrization.data_to_arguments(data, deterministic=False)
        arg_sum += args[0]
        kwarg_sum += kwargs["y"]
    assert arg_sum != 0
    assert kwarg_sum != 0


@testing.parametrized(
    floats=((var.Gaussian(0, 1), var.Array(1).asscalar()), True, False),
    array_int=((var.Gaussian(0, 1), var.Array(1).asscalar(int)), False, False),
    softmax_noisy=((var.SoftmaxCategorical(["blue", "red"]), var.Array(1)), True, True),
    softmax_deterministic=((var.SoftmaxCategorical(["blue", "red"], deterministic=True), var.Array(1)), False, False),
    ordered_discrete=((var.OrderedDiscrete([True, False]), var.Array(1)), False, False),


)
def test_instrumentation_continuous_noisy(variables: tp.Tuple[var.Variable, ...], continuous: bool, noisy: bool) -> None:
    instru = Instrumentation(*variables)
    assert instru.continuous == continuous
    assert instru.noisy == noisy
