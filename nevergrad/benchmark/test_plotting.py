# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import plotting  # pylint: disable=wrong-import-position, wrong-import-order
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')


def test_get_winners_df() -> None:
    data = [["alg0", 46424, .4],
            ["alg1", 4324546, .1],
            ["alg1", 424546, .5],
            ["alg2", 1424546, .3]]
    df = pd.DataFrame(columns=["optimizer_name", "blublu", "loss"], data=data)
    all_optimizers = [f"alg{k}" for k in range(4)]
    # blublu column is useless, and losses are meaned for each algorithm
    winners = plotting._make_winners_df(df, all_optimizers)
    data = .5 * np.identity(4)  # self playing is a tie
    # unspecified algo
    data[(3, 3)] = 0  # type: ignore
    # alg1 and alg2 win over alg0  # type: ignore
    data[tuple(zip(*[(1, 0), (2, 0)]))] = 1  # type: ignore
    # alg1 and alg2 are a tie (mean loss .3)
    data[tuple(zip(*[(1, 2), (2, 1)]))] = .5  # type: ignore
    expected = pd.DataFrame(index=all_optimizers, columns=all_optimizers, data=data)
    assert winners.equals(expected), f"Expected:\n{expected}\nbut got:\n{winners}"


def test_make_sorted_winrates() -> None:
    algos = [f"alg{k}" for k in range(4)]
    data = [[0, 0, 0, 0],  # unplayed
            [0, 2, 4, 4],  # all time winner (4 games
            [0, 0, 2, 1],  # last
            [0, 0, 3, 2]]
    victories = pd.DataFrame(index=algos, columns=algos, data=data)
    winrates = plotting._make_sorted_winrates_df(victories)
    expected_data = [[.5, 1, 1, -1.],
                     [0, .5, .75, -1],
                     [0, .25, .5, -1],
                     [-1, -1, -1, -1]]
    winrates = winrates.fillna(-1)
    salgos = [f"alg{k}" for k in [1, 3, 2, 0]]
    expected = pd.DataFrame(index=salgos, columns=salgos, data=expected_data)
    assert winrates.equals(expected), f"Expected:\n{expected}\nbut got:\n{winrates}"


def test_create_plots_from_csv() -> None:
    df = pd.read_csv(Path(__file__).parent / "sphere_perf_example.csv")
    with patch('matplotlib.pyplot.savefig'):
        plotting.create_plots(df, "", max_combsize=2)


def test_remove_errors() -> None:
    data = [["alg0", 0, 10, np.nan],
            ["alg1", 0, 20, ""],
            ["alg1", np.nan, 30, "ValueError"],
            ["alg2", np.nan, 40, "BlubluError"]]
    df = pd.DataFrame(columns=["optimizer_name", "loss", "dimension", "error"], data=data)
    output = plotting.remove_errors(df)
    expected = pd.DataFrame(columns=["optimizer_name", "loss", "dimension"], data=[["alg0", 0, 10], ["alg1", 0, 20]])
    np.testing.assert_array_equal(output.columns, expected.columns)
    np.testing.assert_array_equal(output.index, expected.index)
    np.testing.assert_array_equal(output, expected)
    assert isinstance(output, plotting.tools.Selector)


def test_make_style_generator() -> None:
    num = 364
    gen = plotting._make_style_generator()
    output = [next(gen) for _ in range(num)]
    np.testing.assert_equal(output[:5], ['-ob', '--vg', ':^r', '-.<c', '->m'])
    # the following is only out of curiosity
    np.testing.assert_equal(len(set(output)), num)  # no repetition
    repeating = next(gen)
    np.testing.assert_equal(repeating, output[0])
