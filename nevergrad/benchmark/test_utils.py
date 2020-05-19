# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import typing as tp
import numpy as np
from nevergrad.common import testing
from . import utils


@testing.parametrized(
    void=({}, ["i1", "i2", "i3"]),
    value=({"c1": "i2-c1"}, ["i2"]),
    function=({"c1": lambda x: x == "i2-c1"}, ["i2"]),
    values=({"c1": ["i3-c1", "i2-c1"]}, ["i2", "i3"]),
    conditions=({"c1": ["i3-c1", "i2-c1"], "c2": "i3-c2"}, ["i3"]),
)
def test_selector(criteria: tp.Any, expected: tp.List[str]) -> None:
    df = utils.Selector(index=["i1", "i2", "i3"], columns=["c1", "c2"])
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
    assert isinstance(df_select, utils.Selector)
    assert isinstance(df_drop, utils.Selector)


@testing.parametrized(
    multiple=(["c1", "c2"], {(2, 1), (2, 2)}),
    unique=(["c1"], {(2,)}),
    unique_2=(["c2"], {(1,), (2,)}),
    none=([], set()),
)
def test_selector_unique(columns: tp.List[str], expected: tp.Iterable[tp.Tuple[int, int]]) -> None:
    df = utils.Selector(index=["i1", "i2", "i3"], columns=["c1", "c2"], data=[[2, 1], [2, 2], [2, 1]])
    testing.printed_assert_equal(df.unique(columns), expected)


def test_selector_assert_equivalent() -> None:
    select1 = utils.Selector(columns=["a", "b"], data=[[0, 1], [2, 3]])
    select2 = utils.Selector(columns=["b", "a"], data=[[3, 2], [1, 0]])
    select3 = utils.Selector(columns=["a", "b"], data=[[0, 5], [2, 3]])
    select1.assert_equivalent(select2)
    np.testing.assert_raises(AssertionError, select1.assert_equivalent, select3)
