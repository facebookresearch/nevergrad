# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import plotting  # pylint: disable=wrong-import-position, wrong-import-order
from unittest.mock import patch
from pathlib import Path
import typing as tp
import pytest
import numpy as np
import pandas as pd
import matplotlib
from nevergrad.common import testing
from . import utils
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
    winners.assert_equivalent(expected)


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
                     [0, 0, 0, -1]]
    winrates = winrates.fillna(-1)
    salgos = [f"alg{k}" for k in [1, 3, 2, 0]]
    expected = pd.DataFrame(index=salgos, columns=salgos, data=expected_data)
    assert winrates.equals(expected), f"Expected:\n{expected}\nbut got:\n{winrates}"


def test_create_plots_from_csv_mocked() -> None:
    df = pd.read_csv(Path(__file__).parent / "sphere_perf_example.csv")
    with patch('nevergrad.benchmark.plotting.XpPlotter'):
        with patch('nevergrad.benchmark.plotting.FightPlotter') as fplt:
            plotting.create_plots(df, "", max_combsize=1)
            assert fplt.call_count == 6, "Should be called for all, 2 noise levels and 3 budgets"


def test_fight_plotter() -> None:
    df = utils.Selector.read_csv(Path(__file__).parent / "sphere_perf_example.csv").select(
        optimizer_name=["OnePlusOneOptimizer", "HaltonSearch", "Powell"])
    winrates = plotting.FightPlotter.winrates_from_selection(df, ["noise_level", "budget"])
    # check data
    np.testing.assert_array_equal(winrates.index, ["Powell (75.0%)", "OnePlusOneOptimizer (58.3%)", "Halton (16.7%)"])
    np.testing.assert_array_equal(winrates.columns, ["Powell (6/6)", "OnePlusOneOptimizer (6/6)", "HaltonSearch (6/6)"])
    np.testing.assert_almost_equal(winrates, [[.5, .75, 1], [.25, .5, 1], [0, 0, .5]])
    # plot
    plotter = plotting.FightPlotter(winrates)
    with patch('matplotlib.pyplot.Figure.savefig'):
        plotter.save("should_not_exist.png")


def test_xp_plotter() -> None:
    opt = "OnePlusOneOptimizer"
    df = utils.Selector.read_csv(Path(__file__).parent / "sphere_perf_example.csv").select(optimizer_name=[opt])
    unused_data = plotting.XpPlotter.make_data(df, normalized_loss=True)
    data = plotting.XpPlotter.make_data(df)
    # check data
    testing.assert_set_equal(data.keys(), {opt})
    testing.assert_set_equal(data[opt].keys(), {"budget", "loss", "loss_std", "num_eval"})
    np.testing.assert_almost_equal(data[opt]["budget"], [200, 400, 800])
    np.testing.assert_almost_equal(data[opt]["loss"], [0.4811605, 0.3920045, 0.14778369])
    np.testing.assert_almost_equal(data[opt]["loss_std"], [0.83034832, 0.73255529, 0.18551625])
    # plot
    with patch('matplotlib.pyplot.Figure.tight_layout'):  # avoid warning message
        plotter = plotting.XpPlotter(data, title="Title")
    with patch('matplotlib.pyplot.Figure.savefig'):
        plotter.save("should_not_exist.png")


def test_remove_errors() -> None:
    data = [["alg0", 0, 10, np.nan],
            ["alg2", np.nan, 30, "ValueError"],
            ["alg1", 0, 20, "SomeHandledError"],
            ["alg3", np.nan, 40, "BlubluError"]]
    df = pd.DataFrame(columns=["optimizer_name", "loss", "dimension", "error"], data=data)
    with pytest.warns(UserWarning) as w:
        output = plotting.remove_errors(df)
    assert len(w) == 3
    expected = pd.DataFrame(columns=["optimizer_name", "loss", "dimension"], data=[["alg0", 0, 10], ["alg1", 0, 20]])
    np.testing.assert_array_equal(output.columns, expected.columns)
    np.testing.assert_array_equal(output.index, expected.index)
    np.testing.assert_array_equal(output, expected)
    assert isinstance(output, plotting.utils.Selector)


def test_remove_nan_value() -> None:
    data = [["alg0", 0, 10, np.nan],
            ["alg2", np.nan, 30, np.nan]]
    df = pd.DataFrame(columns=["optimizer_name", "loss", "dimension", "error"], data=data)
    with pytest.warns(UserWarning) as w:
        output = plotting.remove_errors(df)
    assert len(w) == 1
    expected = pd.DataFrame(columns=["optimizer_name", "loss", "dimension"], data=[["alg0", 0, 10]])
    np.testing.assert_array_equal(output, expected)


def test_make_style_generator() -> None:
    num = 364
    gen = plotting._make_style_generator()
    output = [next(gen) for _ in range(num)]
    np.testing.assert_equal(output[:5], ['-ob', '--vg', ':^r', '-.<c', '->m'])
    # the following is only out of curiosity
    np.testing.assert_equal(len(set(output)), num)  # no repetition
    repeating = next(gen)
    np.testing.assert_equal(repeating, output[0])


def test_name_style() -> None:
    nstyle = plotting.NameStyle()
    np.testing.assert_equal(nstyle["blublu"], "-ob")
    np.testing.assert_equal(nstyle["plop"], "--vg")
    np.testing.assert_equal(nstyle["blublu"], "-ob")


def test_split_long_title() -> None:
    title = "abcd,efgh"
    np.testing.assert_equal(plotting.split_long_title(title), title)
    title = ",".join(["a" * 25, "b" * 25, "c" * 25, "d" * 15])
    np.testing.assert_equal(plotting.split_long_title(title), title[:52] + "\n" + title[52:])
    title = "a" * 70
    np.testing.assert_equal(plotting.split_long_title(title), title)


@testing.parametrized(
    nothing=([1, 2, 10.], [1, 2, 10.]),
    identic=([1, 1, 10., 10.], [.5, 1.5, 9.5, 10.5]),
)
def test_compute_best_placements(positions: tp.List[float], expected: tp.List[float]) -> None:
    new_positions = plotting.compute_best_placements(positions, min_diff=1.)
    np.testing.assert_array_equal(new_positions, expected)


def test_merge_parametrization_and_optimizer() -> None:
    df = pd.DataFrame(
        columns=["optimizer_name", "parametrization", "val"],
        data=[["o1", "p1", 1], ["o1", "p2", 2], ["o2", "p1", 3]]
    )
    out = plotting.merge_parametrization_and_optimizer(utils.Selector(df))
    assert isinstance(out, utils.Selector)
    assert out["optimizer_name"].tolist() == ["o1,p1", "o1,p2", "o2"]
    assert out["val"].tolist() == [1, 2, 3]


def test_normalized_losses() -> None:
    data = [
        ["alg0", 0, "sphere", 3],
        ["alg0", -2, "sphere", 3],
        ["alg2", 2, "sphere", 3],
        ["alg3", 12, "sphere", 12],
        ["alg2", 5, "sphere", 12],
        ["alg4", 24, "ellipsoid", 3],
    ]
    df = pd.DataFrame(columns=["optimizer_name", "loss", "func", "dimension"], data=data)
    ndf = plotting.normalized_losses(df, ["func", "dimension"])
    np.testing.assert_array_equal(ndf.loss, [0.5, 0, 1, 1, 0, 1])


if __name__ == "__main__":
    # simple example which can be run with:
    # python -m nevergrad.benchmark.test_plotting
    df_test = pd.read_csv(Path(__file__).parent / "sphere_perf_example.csv")
    plotting.create_plots(df_test, output_folder="", max_combsize=0)
