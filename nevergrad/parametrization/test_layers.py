# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import numpy as np
import nevergrad as ng
from nevergrad.common import testing
import nevergrad.common.typing as tp
from . import _datalayers
from . import helpers


def test_scalar_module() -> None:
    ref = ng.p.Scalar()
    x = ng.p.Scalar(10) % 4
    assert x.value == 2
    assert x.get_standardized_data(reference=ref)[0] == 10
    x.value = 1
    assert x.get_standardized_data(reference=ref)[0] == 9  # find the closest


def test_bound_module() -> None:
    ref = ng.p.Scalar()
    with pytest.raises(ng.errors.NevergradValueError):
        _datalayers.Bound(3, 8, method="arctan")(ref)
    x = _datalayers.Bound(-1, 8, method="arctan")(ref)
    x.set_standardized_data([100], reference=ref)
    np.testing.assert_almost_equal(x.value, 7.97135306)


def test_log_layer() -> None:
    ref = ng.p.Scalar()
    x = 2 ** ng.p.Scalar()
    assert x.value == 1
    x.value = 16
    assert x.get_standardized_data(reference=ref)[0] == 4  # find the closest
    assert x.value == 16


def test_add_layer() -> None:
    x = ng.p.Scalar() - 4.0
    y = 6 + x
    assert y.value == 2


def test_multiply_layer() -> None:
    ref = ng.p.Scalar()
    x = ng.p.Scalar(5) / 5
    assert x.value == 1
    x.value = 2
    assert x.get_standardized_data(reference=ref)[0] == 10


def test_power() -> None:
    ref = ng.p.Scalar()
    x = 3 / ng.p.Scalar(6)
    assert x.value == 0.5
    x.value = 0.25
    assert x.get_standardized_data(reference=ref)[0] == 12


@testing.parametrized(
    new_unary=(10 ** -ng.p.Scalar(lower=0, upper=5),),
    legacy_log=(ng.p.Log(lower=1e-10, upper=1.0, exponent=5),),
    legacy_array=(
        ng.p.Array(init=1e-5 * np.ones(100)).set_bounds(lower=1e-5, upper=1.0).set_mutation(exponent=10),
    ),
)
def test_log_sampling(log: ng.p.Data) -> None:
    values: tp.List[float] = []
    while len(values) < 100:
        new = log.sample().value
        if isinstance(new, np.ndarray):
            values.extend(new.tolist())
        else:
            values.append(new)
    proba = np.mean(np.array(values) < 0.1)
    assert 0.5 < proba < 1  # should be around 80%


def test_clipping_standardized_data() -> None:
    ref = ng.p.Scalar()
    x = _datalayers.Bound(-10, 10, method="clipping")(ref)
    x.set_standardized_data([12])
    state = x.get_standardized_data(reference=ref)
    assert state[0] == 10
    assert x.value == 10


def test_bound_estimation() -> None:
    param = (_datalayers.Bound(-10, 10)(ng.p.Scalar()) + 3) * 5
    assert param.bounds == (-35, 65)  # type: ignore


def test_softmax_layer() -> None:
    param = ng.p.Array(shape=(4, 3))
    param.random_state.seed(12)
    param.add_layer(_datalayers.SoftmaxSampling(arity=3))
    assert param.value.tolist() == [0, 2, 0, 1]
    assert param.value.tolist() == [0, 2, 0, 1], "Different indices at the second call"
    del param.value
    assert param.value.tolist() == [0, 2, 2, 0], "Same indices after resampling"
    param.value = [0, 1, 2, 0]  # type: ignore
    assert param.value.tolist() == [0, 1, 2, 0]
    expected = np.zeros((4, 3))
    expected[[0, 1, 2, 3], [0, 1, 2, 0]] = 0.6931
    np.testing.assert_array_almost_equal(param._value, expected, decimal=4)


def test_deterministic_softmax_layer() -> None:
    param = ng.p.Array(shape=(1, 100))
    param.add_layer(_datalayers.SoftmaxSampling(arity=100, deterministic=True))
    param._value[0, 12] = 1
    assert param.value.tolist() == [12]


def test_temporary_deterinistic_softmax_layer() -> None:
    param = ng.p.Array(shape=(1, 100))
    param.add_layer(_datalayers.SoftmaxSampling(arity=100, deterministic=False))
    param._value[0, 12] = 1
    del param.value
    with helpers.deterministic_sampling(param):
        assert param.value[0] == 12
    # the behavior should be stochastic when not using the context
    different = False
    for _ in range(10000):
        del param.value
        different = param.value[0] != 12
        if different:
            break
    assert different, "At least one sampling should have been different"


def test_bounded_int_casting() -> None:
    param = _datalayers.Bound(-10.9, 10.9, method="clipping")(ng.p.Scalar())
    param.add_layer(_datalayers.Int())
    for move, val in [(2.4, 2), (0.2, 3), (42, 10), (-42, -10)]:
        param.set_standardized_data([move])
        assert param.value == val, f"Wrong value after move {move}"


def test_rand_int_casting() -> None:
    param = _datalayers.Bound(0, 1)(ng.p.Array(shape=(100, 10)) + 0.2)
    param.add_layer(_datalayers.Int(deterministic=False))
    total = param.value.ravel().sum()
    assert 50 < total < 500


@pytest.mark.parametrize("deg", (True, False))  # type: ignore
@pytest.mark.parametrize("bound_method", (None, "clipping", "arctan"))  # type: ignore
def test_angles(deg: bool, bound_method: tp.Any) -> None:
    span = 360 if deg else 2 * np.pi
    params = [
        _datalayers.Angles(shape=(10,), deg=deg, bound_method=bound_method) / span + 0.5 for _ in range(2)
    ]
    assert params[0].bounds == (0, 1)
    values = [np.linspace(0, 1, 10), np.linspace(1, 0, 10)]
    for param, value in zip(params, values):
        param.value = value
        assert param.value == pytest.approx(value)
    average = params[1].get_standardized_data(reference=params[0]) / 2
    params[1].set_standardized_data(average, reference=params[0])
    assert params[1].value.tolist() == pytest.approx([1, 1, 1, 0.5, 0.5, 0.5, 0.5, 1, 1, 1])
