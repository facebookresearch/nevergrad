# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import numpy as np
from nevergrad.common import testing
from . import mutation
from .data import Array


@testing.parametrized(
    fft=(True, [3, 3, 5, 5]),
    real=(False, [4, 5, 5, 4]),
)
def test_crossover(fft: bool, expected: tp.List[int]) -> None:
    x1 = 4 * np.ones((2, 4))
    x2 = 5 * np.ones((2, 4)) if not fft else np.arange(8).reshape((2, 4))
    co = mutation.Crossover(axis=1, fft=fft)
    co.random_state.seed(12)
    out = co._apply_array((x1, x2))
    expected = np.ones((2, 1)).dot([expected])
    np.testing.assert_array_equal(out, expected)


def test_ravel_crossover() -> None:
    x1 = 4 * np.ones((2, 4))
    x2 = 5 * np.ones((2, 4))
    co = mutation.RavelCrossover().spawn_child()
    co.random_state.seed(12)
    out = co._apply_array((x1, x2))
    expected = [[4, 5, 5, 5], [4, 4, 4, 4]]
    np.testing.assert_array_equal(out, expected)


def test_local_gaussian() -> None:
    init = 4.0 * np.ones((2, 4))
    x = Array(init=np.array(init))
    lg = mutation.LocalGaussian(axes=1, size=2)
    lg.random_state.seed(12)
    lg.apply([x])
    expected = np.ones((2, 1)).dot([[1, 0, 0, 1]])
    np.testing.assert_array_equal(x.value == init, expected)


def test_proba_local_gaussian() -> None:
    init = 4.0 * np.ones((2, 8))
    x = Array(init=np.array(init))
    lg = mutation.ProbaLocalGaussian(axis=1, shape=x.value.shape)
    lg.parameters["ratio"].value = .3
    pattern = [0, 0, 100, 100, 0, 0, 0, 0]
    lg.parameters["positions"].value = pattern
    lg.apply([x])
    expected = np.ones((2, 1)).dot([pattern]) == 0
    np.testing.assert_array_equal(x.value == init, expected)


def test_translation() -> None:
    x = np.arange(4)[:, None].dot(np.ones((1, 2)))
    roll = mutation.Translation(0)
    roll.random_state.seed(12)
    out = roll._apply_array([x])
    expected = np.array([1, 2, 3, 0])[:, None].dot(np.ones((1, 2)))
    np.testing.assert_array_equal(out, expected)
    assert repr(roll) == "Translation[axis=(0,)]"


def test_jump() -> None:
    x = np.arange(6)[:, None].dot(np.ones((1, 2)))
    jump = mutation.Jumping(axis=0, size=5)
    jump.random_state.seed(38)
    out = jump._apply_array([x])
    expected = np.array([0, 3, 4, 1, 2, 5])[:, None].dot(np.ones((1, 2)))
    np.testing.assert_array_equal(out, expected)
    assert repr(jump) == "Jumping[axis=0,size=5]"


def test_tuned_translation() -> None:
    x = np.arange(4)[:, None].dot(np.ones((1, 2)))
    roll = mutation.TunedTranslation(0, shape=x.shape)
    roll.random_state.seed(12)
    out = roll._apply_array([x])
    expected = np.array([3, 0, 1, 2])[:, None].dot(np.ones((1, 2)))
    np.testing.assert_array_equal(out, expected)
    roll.mutate()
    assert np.sum(np.abs(roll.shift.weights.value)) > 0


@testing.parametrized(
    all_none=(None, None),
    d2=((1, 2), None),
    d1=((1), None),
)
def test_crossover_axis(axis: tp.Optional[tp.Tuple[int, ...]], max_size: tp.Optional[int]) -> None:
    shape = (6, 8, 10)
    x1 = 4 * np.ones(shape)
    x2 = 5 * np.ones(shape)
    co = mutation.Crossover(axis=axis, max_size=max_size)
    co.random_state.seed(12)
    out = co._apply_array((x1, x2))
    np.testing.assert_array_equal(out.shape, shape)  # this basically only test that it did not raise an error
    assert co.name.startswith("Crossover[axis="), f"Unexpected {co.name}"
    assert co.name.endswith(f",max_size={max_size}]"), f"Unexpected {co.name}"


@testing.parametrized(
    w1=(1, [1, 2, 5, 9]),
    w2=(2, [3, 7, 14, 10]),
    w3=(3, [8, 16, 15, 12]),
    w4=(4, [17, 17, 17, 17]),
)
def test_rolling_mean(window: int, expected: tp.List[int]) -> None:
    output = mutation.rolling_mean(np.array([1, 2, 5, 9]), window)
    np.testing.assert_array_equal(output, expected)
