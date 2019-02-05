# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import itertools
from pathlib import Path
from typing import Iterator, List, Optional, Any
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
from ..common import tools
from ..common.typetools import PathLike
# pylint: disable=too-many-locals


_DPI = 100


# %% Basic tools

def _make_style_generator() -> Iterator[str]:
    lines = itertools.cycle(["-", "--", ":", "-."])  # 4
    markers = itertools.cycle("ov^<>8sp*hHDd")  # 13
    colors = itertools.cycle("bgrcmyk")  # 7
    return (l + m + c for l, m, c in zip(lines, markers, colors))


class NameStyle(dict):
    """Provides a style for each name, and keeps to it
    """

    def __init__(self) -> None:
        super().__init__()
        self._gen = _make_style_generator()

    def __getitem__(self, name: str) -> Any:
        if name not in self:
            self[name] = next(self._gen)
        return super().__getitem__(name)


def _make_winners_df(df: pd.DataFrame, all_optimizers: List[str]) -> tools.Selector:
    """Finds mean loss over all runs for each of the optimizers, and creates a matrix
    winner_ij = 1 if opt_i is better (lower loss) then opt_j (and .5 for ties)
    """
    if not isinstance(df, tools.Selector):
        df = tools.Selector(df)
    all_optim_set = set(all_optimizers)
    assert all(x in all_optim_set for x in df.unique("optimizer_name"))
    assert all(x in df.columns for x in ["optimizer_name", "loss"])
    winners = tools.Selector(index=all_optimizers, columns=all_optimizers, data=0.)
    grouped = df.loc[:, ["optimizer_name", "loss"]].groupby(["optimizer_name"]).mean()
    df_optimizers = list(grouped.index)
    values = np.array(grouped)
    diffs = values - values.T
    # loss_ij = 1 means opt_i beats opt_j once (beating means getting a lower loss/regret)
    winners.loc[df_optimizers, df_optimizers] = (diffs < 0) + .5 * (diffs == 0)
    return winners


def _make_sorted_winrates_df(victories: pd.DataFrame) -> pd.DataFrame:
    """Converts a dataframe counting number of victories into a sorted
    winrate dataframe. The algorithm which performs better than all other
    algorithms comes first.
    """
    assert all(x == y for x, y in zip(victories.index, victories.columns))
    winrates = victories / (victories + victories.T)
    mean_win = winrates.mean(axis=1).sort_values(ascending=False)
    return winrates.loc[mean_win.index, mean_win.index]


# %% plotting functions

def remove_errors(df: pd.DataFrame) -> tools.Selector:
    df = tools.Selector(df)
    if "error" not in df.columns:  # backward compatibility
        return df  # type: ignore
    # errors with no recommendation
    errordf = df.select(error=lambda x: isinstance(x, str) and x, loss=np.isnan)
    for _, row in errordf.iterrows():
        print(f'Removing "{row["optimizer_name"]}" with dimension {row["dimension"]}: got error "{row["error"]}".')
    # error with recoreded recommendation
    handlederrordf = df.select(error=lambda x: isinstance(x, str) and x, loss=lambda x: not np.isnan(x))
    for _, row in handlederrordf.iterrows():
        print(f'Keeping non-optimal recommendation of "{row["optimizer_name"]}" '
              f'with dimension {row["dimension"]} which raised "{row["error"]}".')
    err_inds = set(errordf.index)
    output = df.loc[[i for i in df.index if i not in err_inds], [c for c in df.columns if c != "error"]]
    assert not output.loc[:, "loss"].isnull().values.any(), "Some nan values remain while there should not be any!"
    output = tools.Selector(output.reset_index(drop=True))
    return output  # type: ignore


def create_plots(df: pd.DataFrame, output_folder: PathLike, max_combsize: int = 1) -> None:
    """Saves all representing plots to the provided folder

    Parameters
    ----------
    df: pd.DataFrame
        the experiment data
    output_folder: PathLike
        path of the folder where the plots should be saved
    max_combsize: int
        maximum number of parameters to fix (combinations) when creating experiment plots
    """
    df = remove_errors(df)
    df.loc[:, "loss"] = pd.to_numeric(df.loc[:, "loss"])
    df = tools.Selector(df.fillna("N-A"))  # remove NaN in non score values
    assert not any("Unnamed: " in x for x in df.columns), f"Remove the unnamed index column:  {df.columns}"
    assert "error " not in df.columns, f"Remove error rows before plotting"
    required = {"optimizer_name", "budget", "loss", "elapsed_time", "elapsed_budget"}
    missing = required - set(df.columns)
    assert not missing, f"Missing fields: {missing}"
    output_folder = Path(output_folder)
    os.makedirs(output_folder, exist_ok=True)
    # check which descriptors do vary
    descriptors = sorted(set(df.columns) - (required | {"seed"}))  # all other columns are descriptors
    to_drop = [x for x in descriptors if len(df.unique(x)) == 1]
    df = tools.Selector(df.loc[:, [x for x in df.columns if x not in to_drop]])
    descriptors = sorted(set(df.columns) - (required | {"seed"}))  # now those should be actual interesting descriptors
    print(f"Descriptors: {descriptors}")
    #
    # fight plot
    # choice of the combination variables to fix
    fight_descriptors = descriptors + ["budget"]  # budget can be used as a descriptor for fight plots
    combinable = [x for x in fight_descriptors if len(df.unique(x)) > 1]  # should be all now
    num_rows = 6
    for fixed in list(itertools.chain.from_iterable(itertools.combinations(combinable, order) for order in range(max_combsize + 1))):
        # choice of the cases with values for the fixed variables
        for case in df.unique(fixed):
            print("\n# new case #", fixed, case)
            casedf = df.select(**dict(zip(fixed, case)))
            name = "fight_" + ",".join("{}{}".format(x, y) for x, y in zip(fixed, case)) + ".png"
            name = "fight_all.png" if name == "fight_.png" else name
            make_fight_plot(casedf, fight_descriptors, num_rows, output_folder / name)
    plt.close("all")
    #
    # xp plots
    # plot mean loss / budget for each optimizer for 1 context
    name_style = NameStyle()  # keep the same style for each algorithm
    for case in df.unique(descriptors):
        subdf = df.select_and_drop(**dict(zip(descriptors, case)))
        description = ",".join("{}:{}".format(x, y) for x, y in zip(descriptors, case))
        out_filepath = output_folder / "xpresults{}{}.png".format("_" if description else "", description.replace(":", ""))
        make_xpresults_plot(subdf, description, out_filepath, name_style)
    plt.close("all")


def make_xpresults_plot(df: pd.DataFrame, title: str, output_filepath: Optional[PathLike] = None,
                        name_style: Optional[dict] = None) -> None:
    """Creates a xp result plot out of the given dataframe: regret with respect to budget for
    each optimizer after averaging on all experiments (it is good practice to use a df
    which is filtered out for one set of input parameters)

    Parameters
    ----------
    df: pd.DataFrame
        run data
    title: str
        title of the plot
    output_filepath: Path
        If present, saves the plot to the given path
    name_style: dict
        a dict or dict-like object providing a line style for each optimizer name.
        (can be helpful for consistency across plots)
    """
    if name_style is None:
        name_style = NameStyle()
    df = tools.Selector(df.loc[:, ["optimizer_name", "budget", "loss"]])
    groupeddf = df.groupby(["optimizer_name", "budget"]).mean()
    groupeddf_std = df.groupby(["optimizer_name", "budget"]).std().loc[groupeddf.index, :]  # std is currently unused
    plt.clf()
    plt.xlabel("Budget")
    plt.ylabel("Loss")
    plt.grid(True, which='both')
    optim_vals = {}
    # extract name and coordinates
    for optim in df.unique("optimizer_name"):
        xvals = np.array(groupeddf.loc[optim, :].index)
        yvals = np.maximum(1e-30, np.array(groupeddf.loc[optim, :].loc[:, "loss"]))  # avoid small vals for logplot
        stds = groupeddf_std.loc[optim, :].loc[:, "loss"]
        optim_name = optim.replace("Search", "").replace("oint", "t").replace("Optimizer", "")
        optim_vals[optim_name] = {"x": xvals, "y": yvals, "std": stds}
    # lower upper bound to twice stupid/idiot at most
    upperbound = max(np.max(vals["y"]) for vals in optim_vals.values())
    for optim, vals in optim_vals.items():
        if optim.lower() in ["stupid", "idiot"] or optim in ["Zero", "StupidRandom"]:
            upperbound = min(upperbound, 2 * np.max(vals["y"]))
    # plot from best to worst
    lowerbound = np.inf
    handles = []
    sorted_optimizers = sorted(optim_vals, key=lambda x: optim_vals[x]["y"][-1], reverse=True)
    for k, optim_name in enumerate(sorted_optimizers):
        vals = optim_vals[optim_name]
        lowerbound = min(lowerbound, np.min(vals["y"]))
        handles.append(plt.loglog(vals["x"], vals["y"], name_style[optim_name], label=optim_name))
        texts = []
        if vals["x"].size and vals["y"][-1] < upperbound:
            angle = 30 - 60 * k / len(optim_vals)
            texts.append(plt.text(vals["x"][-1], vals["y"][-1], "{} ({:.3g})".format(optim_name, vals["y"][-1]),
                                  {'ha': 'left', 'va': 'top' if angle < 0 else 'bottom'}, rotation=angle))
    if upperbound < np.inf:
        plt.gca().set_ylim(lowerbound, upperbound)
    # global info
    legend = plt.legend(fontsize=7, ncol=2, handlelength=3,
                        loc='upper center', bbox_to_anchor=(0.5, -0.15))

    # split long strings at a comma.
    if len(title) > 60:
        indices_commas = [i for i in range(len(title)) if title[i] == ","]
        if len(indices_commas) >= 0:
            min_distance_to_middle = float("inf")
            best_index: Optional[int] = None
            for i in indices_commas:
                distance_to_middle = abs(i - len(title) / 2.)
                if distance_to_middle < min_distance_to_middle:
                    best_index = i
                    min_distance_to_middle = distance_to_middle
            assert best_index is not None
            title = title[:(best_index+1)] + "\n" + title[(best_index+1):]

    plt.title(title)
    # plt.tight_layout()
    # plt.axis('tight')
    # plt.tick_params(axis='both', which='both')
    if output_filepath is not None:
        plt.savefig(str(output_filepath), bbox_extra_artists=[legend] + texts, bbox_inches='tight', dpi=_DPI)


def make_fight_plot(df: tools.Selector, categories: List[str], num_rows: int, output_filepath: Optional[PathLike] = None) -> None:
    """Creates a fight plot out of the given dataframe, by iterating over all cases with fixed category variables.

    Parameters
    ----------
    df: pd.DataFrame
        run data
    categories: list
        List of variables to fix for obtaining similar run conditions
    num_rows: int
        number of rows to plot (best algorithms)
    output_filepath: Path
        If present, saves the plot to the given path
    """
    all_optimizers = list(df.unique("optimizer_name"))  # optimizers for which no run exists are not shown
    num_rows = min(num_rows, len(all_optimizers))
    victories = pd.DataFrame(index=all_optimizers, columns=all_optimizers, data=0.)
    # iterate on all sub cases
    subcases = df.unique(categories)
    for subcase in subcases:  # TODO linearize this (precompute all subcases)? requires memory
        subdf = df.select(**dict(zip(categories, subcase)))
        victories += _make_winners_df(subdf, all_optimizers)
    winrates = _make_sorted_winrates_df(victories)
    mean_win = winrates.mean(axis=1)
    winrates.fillna(.5)  # unplayed
    sorted_names = winrates.index
    # number of subcases actually computed is twice self-victories
    sorted_names = ["{} ({}/{})".format(n, int(2 * victories.loc[n, n]), len(subcases)) for n in sorted_names]
    data = np.array(winrates.iloc[:num_rows, :])
    # make plot
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(100 * data, cmap=cm.seismic, interpolation='none', vmin=0, vmax=100)
    ax.set_xticks(list(range(len(sorted_names))))
    ax.set_xticklabels([s.replace("Search", "") for s in sorted_names], rotation=90, fontsize=7)
    ax.set_yticks(list(range(num_rows)))
    # pylint: disable=anomalous-backslash-in-string
    ax.set_yticklabels([(f"{name} ({100 * val:2.1f}\%)").replace("Search", "") for name, val in zip(mean_win.index[: num_rows], mean_win)], rotation=45, fontsize=7)
    plt.tight_layout()
    fig.colorbar(cax, orientation='vertical')
    if output_filepath is not None:
        plt.savefig(str(output_filepath), dpi=_DPI)


def main() -> None:
    parser = argparse.ArgumentParser(description='Create plots from an experiment data file')
    parser.add_argument('filepath', type=str, help='filepath containing the experiment data')
    parser.add_argument('--output', type=str, default=None,
                        help="Output path for the CSV file (default: a folder <filename>_plots next to the data file.")
    parser.add_argument('--max_combsize', type=int, default=3,
                        help="maximum number of parameters to fix (combinations) when creating experiment plots")
    args = parser.parse_args()
    exp_df = tools.Selector.read_csv(args.filepath)
    output_dir = args.output
    if output_dir is None:
        output_dir = str(Path(args.filepath).with_suffix("")) + "_plots"
    create_plots(exp_df, output_folder=output_dir, max_combsize=args.max_combsize)


if __name__ == '__main__':
    main()
