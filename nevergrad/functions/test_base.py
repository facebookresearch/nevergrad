# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import numpy as np
from nevergrad.parametrization import parameter as p
from nevergrad.common import testing
from . import base


def _arg_return(*args: tp.Any, **kwargs: tp.Any) -> tp.Tuple[tp.Tuple[tp.Any, ...], tp.Dict[str, tp.Any]]:
    return args, kwargs


def test_experiment_function() -> None:
    ifunc = base.ExperimentFunction(_arg_return, p.Instrumentation(  # type: ignore
        p.Choice([1, 12]),
        "constant",
        p.Array(shape=(2, 2)),
        constkwarg="blublu",
        plop=p.Choice([3, 4]),
    ))
    np.testing.assert_equal(ifunc.dimension, 8)
    data = [-100.0, 100, 1, 2, 3, 4, 100, -100]
    args0, kwargs0 = ifunc.parametrization.spawn_child().set_standardized_data(data).value
    output = ifunc(*args0, **kwargs0)  # this is very stupid and should be removed when Parameter is in use
    args: tp.Any = output[0]  # type: ignore
    kwargs: tp.Any = output[1]  # type: ignore
    testing.printed_assert_equal(args, [12, "constant", [[1, 2], [3, 4]]])
    testing.printed_assert_equal(kwargs, {"constkwarg": "blublu", "plop": 3})
    instru_str = ("Instrumentation(Tuple(Choice(choices=Tuple(1,12),"
                  "weights=Array{(1,2)}),constant,"
                  "Array{(2,2)}),"
                  "Dict(constkwarg=blublu,plop=Choice(choices=Tuple(3,4),"
                  "weights=Array{(1,2)})))")
    testing.printed_assert_equal(
        ifunc.descriptors,
        {
            "dimension": 8,
            "name": "_arg_return",
            "function_class": "ExperimentFunction",
            "parametrization": instru_str,
        },
    )


def test_instrumented_function_kwarg_order() -> None:
    ifunc = base.ExperimentFunction(_arg_return, p.Instrumentation(  # type: ignore
        kw4=p.Choice([1, 0]), kw2="constant", kw3=p.Array(shape=(2, 2)), kw1=p.Scalar(2.0).set_mutation(sigma=2.0)
    ))
    np.testing.assert_equal(ifunc.dimension, 7)
    data = np.array([-1, 1, 2, 3, 4, 100, -100])
    args0, kwargs0 = ifunc.parametrization.spawn_child().set_standardized_data(data).value
    # this is very stupid and should be removed when Parameter is in use
    kwargs: tp.Any = ifunc(*args0, **kwargs0)[1]   # type: ignore
    testing.printed_assert_equal(kwargs, {"kw1": 0, "kw2": "constant", "kw3": [[1, 2], [3, 4]], "kw4": 1})


class _Callable:
    def __call__(self, x: float, y: float = 0) -> float:
        return abs(x + y)


def test_callable_parametrization() -> None:
    ifunc = base.ExperimentFunction(lambda x: x ** 2, p.Scalar(2).set_mutation(2))  # type: ignore
    np.testing.assert_equal(ifunc.descriptors["name"], "<lambda>")
    ifunc = base.ExperimentFunction(_Callable(), p.Scalar(2).set_mutation(sigma=2))
    np.testing.assert_equal(ifunc.descriptors["name"], "_Callable")


def test_deterministic_data_setter() -> None:
    instru = p.Instrumentation(p.Choice([0, 1, 2, 3]), y=p.Choice([0, 1, 2, 3]))
    ifunc = base.ExperimentFunction(_Callable(), instru)
    data = [0.01, 0, 0, 0, 0.01, 0, 0, 0]
    for _ in range(20):
        args, kwargs = ifunc.parametrization.spawn_child().set_standardized_data(data, deterministic=True).value
        testing.printed_assert_equal(args, [0])
        testing.printed_assert_equal(kwargs, {"y": 0})
    arg_sum, kwarg_sum = 0, 0
    for _ in range(24):
        args, kwargs = ifunc.parametrization.spawn_child().set_standardized_data(data, deterministic=False).value
        arg_sum += args[0]
        kwarg_sum += kwargs["y"]
    assert arg_sum != 0
    assert kwarg_sum != 0


@testing.parametrized(
    floats=((p.Scalar(), p.Scalar(init=12.0)), True, False),
    array_int=((p.Scalar(), p.Array(shape=(1,)).set_integer_casting()), False, False),
    softmax_noisy=((p.Choice(["blue", "red"]), p.Array(shape=(1,))), True, True),
    softmax_deterministic=((p.Choice(["blue", "red"], deterministic=True), p.Array(shape=(1,))), False, False),
    ordered_discrete=((p.TransitionChoice([True, False]), p.Array(shape=(1,))), False, False),
)
def test_parametrization_continuous_noisy(variables: tp.Tuple[p.Parameter, ...], continuous: bool, noisy: bool) -> None:
    instru = p.Instrumentation(*variables)
    assert instru.descriptors.continuous == continuous
    assert instru.descriptors.deterministic != noisy
