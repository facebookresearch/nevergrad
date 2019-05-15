# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import itertools
from pathlib import Path
from typing import Iterator, List, Optional, Any, Dict, Tuple, NamedTuple
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.legend import Legend
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


class NameStyle(Dict[str, Any]):
    """Provides a style for each name, and keeps to it
    """

    def __init__(self) -> None:
        super().__init__()
        self._gen = _make_style_generator()

    def __getitem__(self, name: str) -> Any:
        if name not in self:
            super().__setitem__(name, next(self._gen))
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
    # mean_win = winrates.quantile(.05, axis=1).sort_values(ascending=False)
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


def create_plots(df: pd.DataFrame, output_folder: PathLike, max_combsize: int = 1, xpaxis: str = "budget") -> None:
    """Saves all representing plots to the provided folder

    Parameters
    ----------
    df: pd.DataFrame
        the experiment data
    output_folder: PathLike
        path of the folder where the plots should be saved
    max_combsize: int
        maximum number of parameters to fix (combinations) when creating experiment plots
    xpaxis: str
        x-axis for xp plots (either budget or pseudotime)
    """
    assert xpaxis in ["budget", "pseudotime"]
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
    descriptors = sorted(set(df.columns) - (required | {"seed", "pseudotime"}))  # all other columns are descriptors
    to_drop = [x for x in descriptors if len(df.unique(x)) == 1]
    df = tools.Selector(df.loc[:, [x for x in df.columns if x not in to_drop]])
    descriptors = sorted(set(df.columns) - (required | {"seed", "pseudotime"}))  # now those should be actual interesting descriptors
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
            data_df = FightPlotter.winrates_from_selection(casedf, fight_descriptors, num_rows=num_rows)
            fplotter = FightPlotter(data_df)
            # save
            name = "fight_" + ",".join("{}{}".format(x, y) for x, y in zip(fixed, case)) + ".png"
            name = "fight_all.png" if name == "fight_.png" else name
            fplotter.save(str(output_folder / name), dpi=_DPI)
    plt.close("all")
    #
    # xp plots
    # plot mean loss / budget for each optimizer for 1 context
    name_style = NameStyle()  # keep the same style for each algorithm
    for case in df.unique(descriptors):
        subdf = df.select_and_drop(**dict(zip(descriptors, case)))
        description = ",".join("{}:{}".format(x, y) for x, y in zip(descriptors, case))
        out_filepath = output_folder / "xpresults{}{}.png".format("_" if description else "", description.replace(":", ""))
        data = XpPlotter.make_data(subdf)
        xpplotter = XpPlotter(data, title=description, name_style=name_style, xaxis=xpaxis)
        xpplotter.save(out_filepath)
    plt.close("all")


class LegendInfo(NamedTuple):
    """Handle for information used to create a legend.
    """
    x: float
    y: float
    line: Any
    text: str


class XpPlotter:
    """Creates a xp result plot out of the given dataframe: regret with respect to budget for
    each optimizer after averaging on all experiments (it is good practice to use a df
    which is filtered out for one set of input parameters)

    Parameters
    ----------
    optim_vals: dict
        output of the make_data static method, containing all information necessary for plotting
    title: str
        title of the plot
    name_style: dict
        a dict or dict-like object providing a line style for each optimizer name.
        (can be helpful for consistency across plots)
    """

    def __init__(self, optim_vals: Dict[str, Dict[str, np.ndarray]], title: str,
                 name_style: Optional[Dict[str, Any]] = None, xaxis: str = "budget") -> None:
        if name_style is None:
            name_style = NameStyle()
        upperbound = max(np.max(vals["loss"]) for vals in optim_vals.values())
        for optim, vals in optim_vals.items():
            if optim.lower() in ["stupid", "idiot"] or optim in ["Zero", "StupidRandom"]:
                upperbound = min(upperbound, 2 * np.max(vals["loss"]))
        # plot from best to worst
        lowerbound = np.inf
        sorted_optimizers = sorted(optim_vals, key=lambda x: optim_vals[x]["loss"][-1], reverse=True)
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111)
        self._ax.autoscale(enable=False)
        self._ax.set_xscale('log')
        self._ax.set_yscale('log')
        self._ax.set_xlabel(xaxis)
        self._ax.set_ylabel("loss")
        self._ax.grid(True, which='both')
        self._overlays: List[Any] = []
        legend_infos: List[LegendInfo] = []
        for optim_name in sorted_optimizers:
            vals = optim_vals[optim_name]
            lowerbound = min(lowerbound, np.min(vals["loss"]))
            line = plt.plot(vals[xaxis], vals["loss"], name_style[optim_name], label=optim_name)
            text = "{} ({:.3g})".format(optim_name, vals["loss"][-1])
            if vals[xaxis].size:
                legend_infos.append(LegendInfo(vals[xaxis][-1], vals["loss"][-1], line, text))
        if upperbound < np.inf:
            self._ax.set_ylim(lowerbound, upperbound)
        all_x = [v for vals in optim_vals.values() for v in vals[xaxis]]
        self._ax.set_xlim([min(all_x), max(all_x)])
        self.add_legends(legend_infos)
        # global info
        self._ax.set_title(split_long_title(title))
        self._ax.tick_params(axis='both', which='both')
        # self._fig.tight_layout()

    def add_legends(self, legend_infos: List[LegendInfo]) -> None:
        """Adds the legends
        """
        # # old way (keep it for fast hacking of plots if need be)
        # # this creates a legend box on the bottom, and algorithm names on the right with some angle to avoid overlapping
        # self._overlays.append(self._ax.legend(fontsize=7, ncol=2, handlelength=3,
        #                                       loc='upper center', bbox_to_anchor=(0.5, -0.2)))
        # upperbound = self._ax.get_ylim()[1]
        # filtered_legend_infos = [i for i in legend_infos if i.y <= upperbound]
        # for k, info in enumerate(filtered_legend_infos):
        #     angle = 30 - 60 * k / len(legend_infos)
        #     self._overlays.append(self._ax.text(info.x, info.y, info.text, {'ha': 'left', 'va': 'top' if angle < 0 else 'bottom'},
        #                                         rotation=angle))
        # new way
        ax = self._ax
        trans = ax.transScale + ax.transLimits
        fontsize = 10.
        display_y = (ax.transAxes.transform((1, 1)) - ax.transAxes.transform((0, 0)))[1]  # height in points
        shift = (2. + fontsize) / display_y
        legend_infos = legend_infos[::-1]  # revert order for use in compute_best_placements
        values = [float(np.clip(trans.transform((0, i.y))[1], -.01, 1.01)) for i in legend_infos]
        placements = compute_best_placements(values, min_diff=shift)
        for placement, info in zip(placements, legend_infos):
            self._overlays.append(Legend(ax, info.line, [info.text], loc="center left",
                                         bbox_to_anchor=(1, placement), frameon=False, fontsize=fontsize))
            ax.add_artist(self._overlays[-1])

    @staticmethod
    def make_data(df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
        """Process raw xp data and process it to extract relevant information for xp plots:
        regret with respect to budget for each optimizer after averaging on all experiments (it is good practice to use a df
        which is filtered out for one set of input parameters)

        Parameters
        ----------
        df: pd.DataFrame
            run data
        xaxis: str
            name of the x-axis among "budget" and  "pseudotime"
        """
        df = tools.Selector(df.loc[:, ["optimizer_name", "budget", "loss"] + (["pseudotime"] if "pseudotime" in df.columns else [])])
        groupeddf = df.groupby(["optimizer_name", "budget"])
        means = groupeddf.mean()
        stds = groupeddf.std()
        optim_vals: Dict[str, Dict[str, np.ndarray]] = {}
        # extract name and coordinates
        for optim in df.unique("optimizer_name"):
            optim_vals[optim] = {}
            optim_vals[optim]["budget"] = np.array(means.loc[optim, :].index)
            optim_vals[optim]["loss"] = np.maximum(1e-30, np.array(means.loc[optim, "loss"]))  # avoid very small values (for log plot)
            optim_vals[optim]["loss_std"] = np.array(stds.loc[optim, "loss"])
            if "pseudotime" in means.columns:
                optim_vals[optim]["pseudotime"] = np.array(means.loc[optim, "pseudotime"])
        return optim_vals

    def save(self, output_filepath: PathLike) -> None:
        """Saves the xp plot

        Parameters
        ----------
        output_filepath: Path or str
            path where the figure must be saved
        """
        self._fig.savefig(str(output_filepath), bbox_extra_artists=self._overlays, bbox_inches='tight', dpi=_DPI)

    def __del__(self) -> None:
        plt.close(self._fig)


def split_long_title(title: str) -> str:
    """Splits a long title around the middle comma
    """
    if len(title) <= 60:
        return title
    comma_indices = np.where(np.array([c for c in title]) == ",")[0]
    if not comma_indices.size:
        return title
    best_index = comma_indices[np.argmin(abs(comma_indices - len(title) // 2))]
    title = title[:(best_index+1)] + "\n" + title[(best_index+1):]
    return title


# @contextlib.contextmanager
# def xticks_on_top() -> Iterator[None]:
#     values_for_top = {'xtick.bottom': False, 'xtick.labelbottom': False,
#                       'xtick.top': True, 'xtick.labeltop': True}
#     defaults = {x: plt.rcParams[x] for x in values_for_top if x in plt.rcParams}
#     plt.rcParams.update(values_for_top)
#     yield
#     plt.rcParams.update(defaults)


class FightPlotter:
    """Creates a fight plot out of the given dataframe, by iterating over all cases with fixed category variables.

    Parameters
    ----------
    winrates_df: pd.DataFrame
        winrate data as a dataframe
    """

    def __init__(self, winrates_df: pd.DataFrame) -> None:
        # make plot
        self.winrates = winrates_df
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111)
        self._cax = self._ax.imshow(100 * np.array(self.winrates), cmap=cm.seismic, interpolation='none', vmin=0, vmax=100)
        x_names = self.winrates.columns
        self._ax.set_xticks(list(range(len(x_names))))
        self._ax.set_xticklabels(x_names, rotation=90, fontsize=7)  # , ha="left")
        y_names = self.winrates.index
        self._ax.set_yticks(list(range(len(y_names))))
        self._ax.set_yticklabels(y_names, rotation=45, fontsize=7)
        self._fig.colorbar(self._cax)  # , orientation='horizontal')
        plt.tight_layout()

    @staticmethod
    def winrates_from_selection(df: tools.Selector, categories: List[str], num_rows: int = 5) -> pd.DataFrame:
        """Creates a fight plot win rate data out of the given run dataframe,
        by iterating over all cases with fixed category variables.

        Parameters
        ----------
        df: pd.DataFrame
            run data
        categories: list
            List of variables to fix for obtaining similar run conditions
        num_rows: int
            number of rows to plot (best algorithms)
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
        # pylint: disable=anomalous-backslash-in-string
        best_names = [(f"{name} ({100 * val:2.1f}%)").replace("Search", "") for name, val in zip(mean_win.index[: num_rows], mean_win)]
        return pd.DataFrame(index=best_names, columns=sorted_names, data=data)

    def save(self, *args: Any, **kwargs: Any) -> None:
        """Shortcut to the figure savefig method
        """
        self._fig.savefig(*args, **kwargs)

    def __del__(self) -> None:
        plt.close(self._fig)


# %% positionning legends

class LegendGroup:
    """Class used to compute legend best placements.
    Each group contains at least one legend, and has a position and span (with bounds). LegendGroup are then
    responsible for providing each of its legends' position (non-overlapping)


    Parameters
    ----------
    indices: List[int]
        identifying index of each of the legends
    init_position: List[float]
        best position for each of the legends (if there was no overlapping)
    min_diff: float
        minimal distance between two legends so that they do not overlap
    """

    def __init__(self, indices: List[int], init_positions: List[float], min_diff: float):
        assert all(x2 - x1 == 1 for x2, x1 in zip(indices[1:], indices[:-1]))
        assert all(v2 >= v1 for v2, v1 in zip(init_positions[1:], init_positions[:-1]))
        assert len(indices) == len(init_positions)
        self.indices = indices
        self.init_positions = init_positions
        self.min_diff = min_diff
        self.position = float(np.mean(init_positions))

    def combine_with(self, other: 'LegendGroup') -> 'LegendGroup':
        assert self.min_diff == other.min_diff
        return LegendGroup(self.indices + other.indices, self.init_positions + other.init_positions, self.min_diff)

    def get_positions(self) -> List[float]:
        first_position = self.bounds[0] + self.min_diff / 2.
        return [first_position + k * self.min_diff for k in range(len(self.indices))]

    @property
    def bounds(self) -> Tuple[float, float]:
        half_span = len(self.indices) * self.min_diff / 2.
        return (self.position - half_span, self.position + half_span)

    def __repr__(self) -> str:
        return f"LegendGroup({self.indices}, {self.init_positions}, {self.min_diff})"


def compute_best_placements(positions: List[float], min_diff: float) -> List[float]:
    """Provides a list of new positions from a list of initial position, with a minimal
    distance between each position.

    Parameters
    ----------
    positions: List[float]
        best positions if minimal distance were 0.
    min_diff: float
        minimal distance allowed between two positions

    Returns
    -------
    new_positions: List[float]
        positions after taking into account the minimal distance constraint

    Note
    ----
    This function is probably not optimal, but seems a very good heuristic
    """
    assert all(v2 >= v1 for v2, v1 in zip(positions[1:], positions[:-1]))
    groups = [LegendGroup([k], [pos], min_diff) for k, pos in enumerate(positions)]
    new_groups: List[LegendGroup] = []
    ready = False
    while not ready:
        ready = True
        for k in range(len(groups)):  # pylint: disable=consider-using-enumerate
            if k < len(groups) - 1 and groups[k + 1].bounds[0] < groups[k].bounds[1]:
                # groups are overlapping: create a new combined group
                # which will provide new non-overlapping positions around the mean of initial positions
                new_groups.append(groups[k].combine_with(groups[k + 1]))
                # copy the rest of the groups and start over from the first group
                new_groups.extend(groups[k + 2:])
                groups = new_groups
                new_groups = []
                ready = False
                break
            else:
                new_groups.append(groups[k])
    new_positions = np.array(positions, copy=True)
    for group in groups:
        new_positions[group.indices] = group.get_positions()
    return new_positions.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description='Create plots from an experiment data file')
    parser.add_argument('filepath', type=str, help='filepath containing the experiment data')
    parser.add_argument('--output', type=str, default=None,
                        help="Output path for the CSV file (default: a folder <filename>_plots next to the data file.")
    parser.add_argument('--max_combsize', type=int, default=0,
                        help="maximum number of parameters to fix (combinations) when creating experiment plots")
    parser.add_argument('--pseudotime', nargs="?", default=False, const=True,
                        help="Plots with respect to pseudotime instead of budget")
    args = parser.parse_args()
    exp_df = tools.Selector.read_csv(args.filepath)
    output_dir = args.output
    if output_dir is None:
        output_dir = str(Path(args.filepath).with_suffix("")) + "_plots"
    create_plots(exp_df, output_folder=output_dir, max_combsize=args.max_combsize, xpaxis="pseudotime" if args.pseudotime else "budget")


if __name__ == '__main__':
    main()
