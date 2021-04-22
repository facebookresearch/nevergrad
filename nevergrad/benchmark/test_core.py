# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=wrong-import-position, wrong-import-order
from .__main__ import repeated_launch
import sys
import warnings
import tempfile
import itertools
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
from nevergrad.optimization import optimizerlib
from nevergrad.parametrization.utils import CommandFunction, FailedJobError
from nevergrad.common import testing
from nevergrad.common import errors
from . import utils
from . import core
from .test_xpbase import DESCRIPTION_KEYS

matplotlib.use("Agg")


@testing.parametrized(
    val0=(0, False),
    val1=(1, True),
    val5=(5, False),
    val6=(6, True),
)
def test_moduler(value: int, expected: bool) -> None:
    moduler = core.Moduler(5, 1)
    np.testing.assert_equal(moduler(value), expected)


def test_compute() -> None:
    output = core.compute("basic")
    assert isinstance(output, utils.Selector)


def test_commandline_launch() -> None:
    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder) / "benchmark_launch_test.csv"
        # commandline test
        # TODO make it work on Windows!
        # TODO make it work again on the CI (Linux), this started failing with #630 for no reason
        with testing.skip_error_on_systems(FailedJobError, systems=("Windows", "Linux")):
            CommandFunction(
                command=[
                    sys.executable,
                    "-m",
                    "nevergrad.benchmark",
                    "additional_experiment",
                    "--cap_index",
                    "2",
                    "--num_workers",
                    "2",
                    "--output",
                    str(output),
                    "--imports",
                    str(Path(__file__).parent / "additional" / "example.py"),
                ]
            )()
        assert output.exists()
        df = utils.Selector.read_csv(str(output))
        testing.assert_set_equal(
            df.columns, DESCRIPTION_KEYS | {"offset"}
        )  # "offset" comes from the custom function
        np.testing.assert_equal(len(df), 2)


def test_launch() -> None:
    with tempfile.TemporaryDirectory() as folder:
        with patch("nevergrad.benchmark.plotting.create_plots"):  # dont plot
            output = Path(folder) / "benchmark_launch_test.csv"
            # commandline test
            repeated_launch("repeated_basic", cap_index=4, num_workers=2, output=output, plot=True)
            assert output.exists()
            df = utils.Selector.read_csv(str(output))
            testing.assert_set_equal(df.unique("optimizer_name"), {"DifferentialEvolution()", "OnePlusOne"})
            assert isinstance(df, utils.Selector)
            np.testing.assert_equal(len(df), 4)


def test_import_additional_module() -> None:
    optim_name = "NewOptimizer"
    xp_name = "additional_experiment"
    assert optim_name not in optimizerlib.registry
    assert xp_name not in core.registry
    core.import_additional_module(Path(__file__).parent / "additional" / "example.py")
    assert optim_name in optimizerlib.registry
    assert xp_name in core.registry


def test_save_or_append_to_csv() -> None:
    with tempfile.TemporaryDirectory() as folder:
        csvpath = Path(folder) / "test_file.csv"
        df = pd.DataFrame(columns=["a", "b"], data=[[1, 2], [3, 4]])
        core.save_or_append_to_csv(df, csvpath)
        df = pd.DataFrame(columns=["a", "c"], data=[[5, 6]])
        core.save_or_append_to_csv(df, csvpath)
        df = pd.read_csv(csvpath).fillna(-1)
        np.testing.assert_array_equal(df.columns, ["a", "b", "c"])
        np.testing.assert_array_equal(np.array(df), [[1, 2, -1], [3, 4, -1], [5, -1, 6]])


def test_moduler_split() -> None:
    total_length = np.random.randint(100, 200)
    split = np.random.randint(1, 12)
    modulers = core.Moduler(1, 0, total_length).split(split)
    data = list(range(total_length))
    err_msg = f"Moduler failed for total_length={total_length} and split={split}."
    all_indices = set()
    for moduler in modulers:
        indices = {k for k in data if moduler(k)}
        np.testing.assert_equal(len(indices), len(moduler), err_msg=err_msg)
        all_indices.update(indices)
    np.testing.assert_equal(len(all_indices), len(data), err_msg=err_msg)


def test_experiment_chunk_split() -> None:
    chunk = core.BenchmarkChunk(name="repeated_basic", seed=12, repetitions=2)
    chunks = chunk.split(2)
    chunks = [chunks[0]] + chunks[1].split(3)
    np.testing.assert_array_equal([len(c) for c in chunks], [10, 4, 3, 3])
    chained = [x[0] for x in itertools.chain.from_iterable(chunks)]
    # check full order (everythink only once)
    np.testing.assert_array_equal(
        chained, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 7, 13, 19, 3, 9, 15, 5, 11, 17]
    )
    testing.assert_set_equal(chained, range(20))
    assert chunks[0].id.endswith("_i0m2"), f"Wrong id {chunks[0].id}"
    assert chunks[0].id[:-4] == chunk.id[:-4], "Id prefix should be inherited"


def test_experiment_chunk_seeding() -> None:
    cap_index = 2
    chunk = core.BenchmarkChunk(name="repeated_basic", seed=12, repetitions=2, cap_index=cap_index)
    xps = [xp for _, xp in chunk]
    assert xps[0].seed != xps[cap_index].seed


def test_benchmark_chunk_resuming() -> None:
    chunk = core.BenchmarkChunk(name="repeated_basic", seed=12, repetitions=1, cap_index=2)
    # creating an error on the first experiment
    with patch("nevergrad.benchmark.xpbase.Experiment.run") as run:
        run.side_effect = ValueError("test error string")
        np.testing.assert_raises(ValueError, chunk.compute)
    # making sure we restart from the actual experiment
    assert chunk._current_experiment is not None
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings("ignore", category=errors.InefficientSettingsWarning)
        chunk.compute()
        assert (
            not w
        ), f"A warning was raised while it should not have (experiment could not be resumed): {w[0].message}"
