# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This test module contains old tests coming from the instrumentation package.
Some are still relevant, others are not, we should sort this out eventually.
Overall, they may be overly complicated because they were converted from the old framework...
"""

import typing as tp
import numpy as np
import pytest
from . import parameter as p


def test_instrumentation_set_standardized_data() -> None:
    tokens = [p.Choice(list(range(5))), p.Scalar(init=3).set_mutation(sigma=4)]
    instru = p.Instrumentation(*tokens)
    values = instru.spawn_child().set_standardized_data([0, 200, 0, 0, 0, 2]).args
    assert values == (1, 11)
    np.testing.assert_raises(
        ValueError, instru.spawn_child().set_standardized_data, [0, 0, 200, 0, 0, 0, 2, 3]
    )


def test_instrumentation() -> None:
    instru = p.Instrumentation(p.Scalar(), 3, b=p.Choice([0, 1, 2, 3]), a=p.TransitionChoice([0, 1, 2, 3]))
    np.testing.assert_equal(instru.dimension, 6)
    instru2 = p.Instrumentation(p.Scalar(), 3, b=p.Choice([0, 1, 2, 3]), a=p.TransitionChoice([0, 1, 2, 3]))
    np.testing.assert_equal(instru2.dimension, 6)
    data = instru2.spawn_child(new_value=((4, 3), dict(a=0, b=3))).get_standardized_data(reference=instru2)
    np.testing.assert_array_almost_equal(data, [4, -1.1503, 0, 0, 0, 0.5878], decimal=4)
    child = instru.spawn_child()
    with p.helpers.deterministic_sampling(child):
        args, kwargs = child.set_standardized_data(data).value
    assert (args, kwargs) == ((4.0, 3), {"a": 0, "b": 3})
    assert "3),Dict(a=TransitionChoice(choices=Tuple(0,1,2,3)," in repr(
        instru
    ), f"Erroneous representation {instru}"
    # check deterministic
    data = np.array([0.0, 0, 0, 0, 0, 0])
    total = 0
    for _ in range(24):
        child = instru.spawn_child()
        with p.helpers.deterministic_sampling(child):
            total += child.set_standardized_data(data).kwargs["b"]
    np.testing.assert_equal(total, 0)
    # check stochastic
    for _ in range(24):
        total += instru.spawn_child().set_standardized_data(data).kwargs["b"]
    assert total != 0
    # check duplicate
    # instru2 = mvar.Instrumentation(*instru.args, **instru.kwargs)  # TODO: OUCH SILENT FAIL
    instru2.copy()
    data = np.random.normal(0, 1, size=6)
    values: tp.List[tp.Any] = []
    for val_instru in [instru, instru2]:
        child = val_instru.spawn_child()
        with p.helpers.deterministic_sampling(child):
            values.append(child.set_standardized_data(data).value)
    assert values[0] == values[1]
    # check naming
    instru_str = (
        "Instrumentation(Tuple(Scalar[sigma=Scalar{exp=2.03}],3),"
        "Dict(a=TransitionChoice(choices=Tuple(0,1,2,3),"
        "indices=Array{Cd(0,4),Add,Int},transitions=[1. 1.]),"
        "b=Choice(choices=Tuple(0,1,2,3),indices=Array{(1,4),SoftmaxSampling})))"
    )
    assert instru.name == instru_str
    assert instru.set_name("blublu").name == "blublu"


def _false(value: tp.Any) -> bool:  # pylint: disable=unused-argument
    return False


def test_instrumentation_copy() -> None:
    instru = p.Instrumentation(p.Scalar(), 3, b=p.Choice(list(range(1000)))).set_name("bidule")
    instru.register_cheap_constraint(_false)
    copied = instru.copy()
    assert copied.name == "bidule"
    assert copied.random_state is not instru.random_state
    # test that variables do not hold a random state / interfere
    instru.random_state = np.random.RandomState(12)
    copied.random_state = np.random.RandomState(12)
    kwargs1 = instru.spawn_child().set_standardized_data([0] * 1001).kwargs
    kwargs2 = copied.spawn_child().set_standardized_data([0] * 1001).kwargs
    assert kwargs1 == kwargs2
    assert not copied.satisfies_constraints()


def test_instrumentation_init_error() -> None:
    variable = p.Scalar()
    np.testing.assert_raises(ValueError, p.Instrumentation, variable, variable)


def test_softmax_categorical_deterministic() -> None:
    token = p.Choice(["blu", "blublu", "blublublu"], deterministic=True)
    assert token.set_standardized_data([1, 1, 1.01]).value == "blublublu"


def test_softmax_categorical() -> None:
    np.random.seed(12)
    token = p.Choice(["blu", "blublu", "blublublu"])
    assert token.spawn_child().set_standardized_data([0.5, 1.0, 1.5]).value == "blublu"
    new_token = token.spawn_child(new_value="blu")
    child = token.spawn_child()
    with p.helpers.deterministic_sampling(child):
        value = child.set_standardized_data(new_token.get_standardized_data(reference=token)).value
    assert value == "blu"


def test_ordered_discrete() -> None:
    token = p.TransitionChoice(["blu", "blublu", "blublublu"])
    assert token.spawn_child().set_standardized_data([5]).value == "blublublu"
    assert token.spawn_child().set_standardized_data([0]).value == "blublu"
    new_token = token.spawn_child(new_value="blu")
    child = token.spawn_child()
    with p.helpers.deterministic_sampling(child):
        value = child.set_standardized_data(new_token.get_standardized_data(reference=token)).value
    assert value == "blu"


def test_scalar() -> None:
    token = p.Scalar().set_integer_casting()
    assert token.spawn_child().set_standardized_data([0.7]).value == 1
    new_token = token.spawn_child(new_value=1)
    assert new_token.get_standardized_data(reference=token).tolist() == [1.0]


# bouncing with large values clips to the other side
@pytest.mark.parametrize("value,expected", [(0, 0.01), (10, 0.001), (-30, 0.1), (20, 0.001)])  # type: ignore
def test_log(value: float, expected: float) -> None:
    var = p.Log(lower=0.001, upper=0.1)
    out = var.spawn_child().set_standardized_data(np.array([value]))
    np.testing.assert_approx_equal(out.value, expected, significant=4)
    repr(var)


def test_log_int() -> None:
    var = p.Log(lower=300, upper=10000).set_integer_casting()
    out = var.spawn_child().set_standardized_data(np.array([0]))
    assert out.value == 1732


# note: 0.9/0.9482=0.9482/0.999
# with very large values, bouncing clips to the other side
@pytest.mark.parametrize("value,expected", [(0, 0.9482), (-11, 0.999), (10, 0.9)])  # type: ignore
def test_log_9(value: float, expected: float) -> None:
    var = p.Log(lower=0.9, upper=0.999)
    out = var.spawn_child().set_standardized_data(np.array([value]))
    np.testing.assert_approx_equal(out.value, expected, significant=4)
