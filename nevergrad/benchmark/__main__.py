# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path
from concurrent import futures
import nevergrad.common.typing as tp
from . import utils
from . import core
from . import plotting


# pylint: disable=too-many-arguments
def launch(
    experiment: str,
    num_workers: int = 1,
    seed: tp.Optional[int] = None,
    cap_index: tp.Optional[int] = None,
    output: tp.Optional[tp.PathLike] = None,
) -> Path:
    """Launch experiment with given names and selection modulo
    max_index can be specified to provide a limited number of settings
    """
    # create the data
    csvpath = Path(experiment + ".csv") if output is None else Path(output)
    if num_workers == 1:
        df = core.compute(experiment, cap_index=cap_index, seed=seed)
    else:
        with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            df = core.compute(
                experiment, seed=seed, cap_index=cap_index, executor=executor, num_workers=num_workers
            )
    # save data to csv
    try:
        core.save_or_append_to_csv(df, csvpath)
    except Exception:  # pylint: disable=broad-except
        csvpath = Path(experiment + ".csv")
        print(f"Failed to save to {output}, falling back to {csvpath}")
        core.save_or_append_to_csv(df, csvpath)
    else:
        print(f"Saved data to {csvpath}")
    return csvpath


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an experiment and create a result csv file.")
    parser.add_argument(
        "experiment", type=str, help="name of an experiment registered in the experiments registry"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Use a seed for reproducibility (for generators which take care of seeding)",
    )
    parser.add_argument(
        "--cap_index", type=int, default=None, help="Stop after generationg/running settings #cap_index"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for the CSV file (default: <experiment>.csv). Existing files are appended",
    )
    parser.add_argument(
        "--imports",
        type=str,
        default=None,
        help="Comma-separated list of file paths with additional experiment(s) and/or optimizer(s) definitions",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Numbers of workers to use for the computation (splits the job in chunks)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Number of repetitions to perform for the experiment plan (seeds will be incremented)",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        default=False,
        const=True,
        help="Creates the corresponding plots if present (provide a path, or folder <experiment>_plots will be used)",
    )
    return parser.parse_args()


def repeated_launch(
    experiment: str,
    num_workers: int = 1,
    seed: tp.Optional[int] = None,
    cap_index: tp.Optional[int] = None,
    output: tp.Optional[tp.PathLike] = None,
    plot: tp.Union[bool, tp.PathLike] = False,
    imports: tp.Optional[tp.List[tp.PathLike]] = None,
    repetitions: int = 1,
) -> None:
    """Launch experiment with given names and selection module
    max_index can be specified to provide a limited number of settings
    This repeats the plan several times and increments the seed.
    """
    # start by importing additional content
    if imports is not None:
        assert isinstance(imports, (tuple, list))
        for path in imports:
            core.import_additional_module(path)
    # then run multiple times
    csvpath = Path("default.csv")
    for k in range(repetitions):
        print(f"Starting repetition {k +1} / {repetitions}")
        csvpath = launch(
            experiment,
            num_workers=num_workers,
            cap_index=cap_index,
            output=output,
            seed=None if seed is None else seed + k,
        )
    # save plots if need be
    if plot:
        df = utils.Selector.read_csv(csvpath)
        if isinstance(plot, bool):
            plot = str(Path(csvpath).with_suffix("")) + "_plots"
        print(f"Saving plots into folder: {plot}")
        plotting.create_plots(df, output_folder=plot)


if __name__ == "__main__":
    args = get_args()
    repeated_launch(
        args.experiment,
        num_workers=args.num_workers,
        cap_index=args.cap_index,
        output=args.output,
        seed=args.seed,
        plot=args.plot,
        imports=args.imports if args.imports is None else args.imports.split(","),
        repetitions=args.repetitions,
    )
