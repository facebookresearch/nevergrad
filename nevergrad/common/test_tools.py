# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from typing import Iterable, List, Any, Tuple
import numpy as np
from . import tools
from . import testing


@testing.parametrized(
    void=([], []),
    one=(["a"], []),
    two=([1, 2], [(1, 2)]),
    three=([1, 2, 3], [(1, 2), (2, 3)]),
)
def test_pairwise(iterator: Iterable[Any], expected: List[Tuple[Any, ...]]) -> None:
    output = list(tools.pairwise(iterator))
    testing.printed_assert_equal(output, expected)


@testing.parametrized(
    void=({}, ["i1", "i2", "i3"]),
    value=({"c1": "i2-c1"}, ["i2"]),
    function=({"c1": lambda x: x == "i2-c1"}, ["i2"]),
    values=({"c1": ["i3-c1", "i2-c1"]}, ["i2", "i3"]),
    conditions=({"c1": ["i3-c1", "i2-c1"], "c2": "i3-c2"}, ["i3"]),
)
def test_selector(criteria: Any, expected: List[str]) -> None:
    df = tools.Selector(index=["i1", "i2", "i3"], columns=["c1", "c2"])
    for i, c in itertools.product(df.index, df.columns):
        df.loc[i, c] = f"{i}-{c}"
    df_select = df.select(**criteria)
    df_drop = df.select_and_drop(**criteria)
    # indices
    testing.assert_set_equal(df_select.index, expected)
    testing.assert_set_equal(df_drop.index, expected)
    # columns
    testing.assert_set_equal(df_select.columns, df)
    testing.assert_set_equal(df_drop.columns, set(df_select.columns) - set(criteria))
    # values
    for i, c in itertools.product(df_select.index, df_select.columns):
        assert df.loc[i, c] == f"{i}-{c}", "Erroneous values"
    # instance
    assert isinstance(df_select, tools.Selector)
    assert isinstance(df_drop, tools.Selector)


def test_roundrobin() -> None:
    output = list(tools.roundrobin([1, 2, 3], (x for x in [4, 5, 6, 7]), (8,)))
    np.testing.assert_array_equal(output, [1, 4, 8, 2, 5, 3, 6, 7])


@testing.parametrized(
    multiple=(["c1", "c2"], {(2, 1), (2, 2)}),
    unique=(["c1"], {(2,)}),
    unique_2=(["c2"], {(1,), (2,)}),
    none=([], set()),
)
def test_selector_unique(columns: List[str], expected: Iterable[Tuple[int, int]]) -> None:
    df = tools.Selector(index=["i1", "i2", "i3"], columns=["c1", "c2"], data=[[2, 1], [2, 2], [2, 1]])
    testing.printed_assert_equal(df.unique(columns), expected)


def test_grouper() -> None:
    output = list(tools.grouper('ABCDEFG', 3, 'x'))
    testing.printed_assert_equal(output, [list(x) for x in ["ABC", "DEF", "Gxx"]])


def test_selector_assert_equivalent() -> None:
    select1 = tools.Selector(columns=["a", "b"], data=[[0, 1], [2, 3]])
    select2 = tools.Selector(columns=["b", "a"], data=[[3, 2], [1, 0]])
    select3 = tools.Selector(columns=["a", "b"], data=[[0, 5], [2, 3]])
    select1.assert_equivalent(select2)
    np.testing.assert_raises(AssertionError, select1.assert_equivalent, select3)


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
