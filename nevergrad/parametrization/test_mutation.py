# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import numpy as np
from nevergrad.common import testing
from . import mutation


def test_crossover() -> None:
    x1 = 4 * np.ones((2, 3))
    x2 = 5 * np.ones((2, 3))
    co = mutation.Crossover(axis=1)
    out = co._apply_array((x1, x2), rng=np.random.RandomState(12))
    expected = np.ones((2, 1)).dot([[4, 5, 4]])
    np.testing.assert_array_equal(out, expected)


def test_rolling() -> None:
    x = np.arange(4)[:, None].dot(np.ones((1, 2)))
    roll = mutation.Translation(0)
    out = roll._apply_array([x], rng=np.random.RandomState(12))
    expected = np.array([1, 2, 3, 0])[:, None].dot(np.ones((1, 2)))
    np.testing.assert_array_equal(out, expected)
    assert repr(roll) == "Translation(axis=(0,))"


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
    out = co._apply_array((x1, x2), rng=np.random.RandomState(12))
    np.testing.assert_array_equal(out.shape, shape)  # this basically only test that it did not raise an error
    assert co.name.startswith("Crossover(axis="), f"Unexpected {co.name}"
    assert co.name.endswith(f")[max_size={max_size}]"), f"Unexpected {co.name}"
