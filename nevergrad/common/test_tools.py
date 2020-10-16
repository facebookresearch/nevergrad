# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import numpy as np
from . import tools
from . import testing


@testing.parametrized(
    void=([], []),
    one=(["a"], []),
    two=([1, 2], [(1, 2)]),
    three=([1, 2, 3], [(1, 2), (2, 3)]),
)
def test_pairwise(iterator: tp.Iterable[tp.Any], expected: tp.List[tp.Tuple[tp.Any, ...]]) -> None:
    output = list(tools.pairwise(iterator))
    testing.printed_assert_equal(output, expected)


def test_roundrobin() -> None:
    output = list(tools.roundrobin([1, 2, 3], (x for x in [4, 5, 6, 7]), (8,)))
    np.testing.assert_array_equal(output, [1, 4, 8, 2, 5, 3, 6, 7])


def test_grouper() -> None:
    output = list(tools.grouper('ABCDEFG', 3, 'x'))
    testing.printed_assert_equal(output, [list(x) for x in ["ABC", "DEF", "Gxx"]])


def test_sleeper() -> None:
    min_sleep = 1e-5
    sleeper = tools.Sleeper(min_sleep=min_sleep)
    np.testing.assert_almost_equal(sleeper._get_advised_sleep_duration(), min_sleep, decimal=5)
    sleeper.start_timer()
    # not precise enough to test for exact time
    # np.testing.assert_almost_equal(sleeper._get_advised_sleep_duration(), min_sleep, decimal=5)
    sleeper.stop_timer()
    # np.testing.assert_almost_equal(sleeper._get_advised_sleep_duration(), min_sleep, decimal=5)


def test_mutable_set() -> None:
    s: tools.OrderedSet[int] = tools.OrderedSet((1, 2, 3))
    assert tuple(s) == (1, 2, 3)
    s.add(1)
    s.add(4)
    assert tuple(s) == (2, 3, 1, 4)
    #
    union = s & {3, 2}
    assert isinstance(union, tools.OrderedSet)
    assert tuple(union) == (2, 3)
    #
    assert s.pop() == 2
    assert s.popright() == 4
    assert tuple(s) == (3, 1)
    #
    s = tools.OrderedSet((1, 2, 3))
    intersect = s | {5, 6}
    assert isinstance(intersect, tools.OrderedSet)
    assert tuple(intersect) == (1, 2, 3, 5, 6)
    intersect = {5, 6} | s
    assert isinstance(intersect, tools.OrderedSet)
    assert tuple(intersect) == (1, 2, 3, 5, 6)  # same behavior, always appended to the end
