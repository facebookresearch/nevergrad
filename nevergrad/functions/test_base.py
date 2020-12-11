# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import numpy as np
import pytest
from nevergrad.parametrization import parameter as p
from nevergrad.common import testing
import nevergrad.common.typing as tp
from . import base


def _arg_return(*args: tp.Any, **kwargs: tp.Any) -> float:
    # for the sake of tests just do some really stupid return
    return args, kwargs  # type: ignore


def test_experiment_function() -> None:
    param = p.Instrumentation(
        p.Choice([1, 12]),
        "constant",
        p.Array(shape=(2, 2)),
        constkwarg="blublu",
        plop=p.Choice([3, 4]),
    )
    with pytest.raises(RuntimeError):
        base.ExperimentFunction(_arg_return, param)
    param.set_name("myparam")
    ifunc = base.ExperimentFunction(_arg_return, param)
    np.testing.assert_equal(ifunc.dimension, 8)
    data = [-100.0, 100, 1, 2, 3, 4, 100, -100]
    args0, kwargs0 = ifunc.parametrization.spawn_child().set_standardized_data(data).value
    output: tp.Any = ifunc(*args0, **kwargs0)
    args: tp.Any = output[0]
    kwargs: tp.Any = output[1]
    testing.printed_assert_equal(args, [12, "constant", [[1, 2], [3, 4]]])
    testing.printed_assert_equal(kwargs, {"constkwarg": "blublu", "plop": 3})
    testing.printed_assert_equal(
        ifunc.descriptors,
        {
            "dimension": 8,
            "name": "_arg_return",
            "function_class": "ExperimentFunction",
            "parametrization": "myparam"
        },
    )


def test_array_experiment_function() -> None:
    iarrayfunc = base.ArrayExperimentFunction(lambda x: sum(x),   # type: ignore
        p.Array(shape=(10,)).set_bounds(-0.5, 6.).set_name(""),
        symmetry=247,
    )
    iarrayfunc2 = base.ArrayExperimentFunction(lambda x: sum(x),   # type: ignore
        p.Array(shape=(10,)).set_bounds(-0.5, 6.).set_name(""),
        symmetry=111,
    )
    np.testing.assert_equal(iarrayfunc.dimension, 10)
    assert iarrayfunc(np.zeros(10)) == iarrayfunc.copy()(np.zeros(10))
    assert iarrayfunc(np.zeros(10)) == 8.25
    assert iarrayfunc(np.ones(10)) != iarrayfunc2(np.ones(10))


def test_instrumented_function_kwarg_order() -> None:
    ifunc = base.ExperimentFunction(_arg_return, p.Instrumentation(
        kw4=p.Choice([1, 0]), kw2="constant", kw3=p.Array(shape=(2, 2)), kw1=p.Scalar(2.0).set_mutation(sigma=2.0)
    ).set_name("test"))
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
    ifunc = base.ExperimentFunction(lambda x: x ** 2, p.Scalar(2).set_mutation(2).set_name(""))  # type: ignore
    np.testing.assert_equal(ifunc.descriptors["name"], "<lambda>")
    ifunc = base.ExperimentFunction(_Callable(), p.Scalar(2).set_mutation(sigma=2).set_name(""))
    np.testing.assert_equal(ifunc.descriptors["name"], "_Callable")
    # test automatic filling
    assert len(ifunc._auto_init) == 2


def test_packed_function() -> None:
    ifunc = base.ExperimentFunction(_Callable(), p.Scalar(1).set_name(""))
    with pytest.raises(AssertionError):
        base.MultiExperiment([ifunc, ifunc], [100, 100])
    pfunc = base.MultiExperiment([ifunc, ifunc.copy()], [100, 100])
    np.testing.assert_equal(pfunc.descriptors["name"], "_Callable,_Callable")
    np.testing.assert_array_equal(pfunc(-3), [3, 3])


def test_deterministic_data_setter() -> None:
    instru = p.Instrumentation(p.Choice([0, 1, 2, 3]), y=p.Choice([0, 1, 2, 3])).set_name("")
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


class ExampleFunction(base.ExperimentFunction):

    def __init__(self, dimension: int, number: int, default: int = 12):  # pylint: disable=unused-argument
        # unused argument is used to check that it is automatically added as descriptor
        super().__init__(self.oracle_call, p.Array(shape=(dimension,)))

    def oracle_call(self, x: np.ndarray) -> float:
        return float(x[0])

    # pylint: disable=unused-argument
    def compute_pseudotime(self, input_parameter: tp.Any, loss: tp.Loss) -> float:
        assert isinstance(loss, (int, float))
        return 5 - loss


def test_function_descriptors_and_pickle() -> None:
    func = ExampleFunction(dimension=1, number=3)
    assert "default" in func.descriptors
    assert "self" not in func._auto_init
    out = pickle.dumps(func)
    func2 = pickle.loads(out).copy()
    assert func2.descriptors["number"] == 3


class ExampleFunctionAllDefault(base.ExperimentFunction):

    def __init__(self, dimension: int = 2, default: int = 12):  # pylint: disable=unused-argument
        # unused argument is used to check that it is automatically added as descriptor
        super().__init__(lambda x: 3.0, p.Array(shape=(dimension,)))


def test_function_descriptors_all_default() -> None:
    func = ExampleFunctionAllDefault()
    assert func.descriptors["default"] == 12
    assert "self" not in func._auto_init
    with pytest.raises(TypeError):
        # make sure unexpected keyword still works
        ExampleFunctionAllDefault(blublu=12)   # type: ignore
