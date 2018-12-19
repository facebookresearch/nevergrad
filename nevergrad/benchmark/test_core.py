# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import itertools
from unittest import TestCase
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
import genty
import matplotlib
from ..optimization import optimizerlib
from ..instrumentation.utils import CommandFunction
from ..common import testing
from . import core
from .test_xpbase import DESCRIPTION_KEYS
matplotlib.use('Agg')
# pylint: disable=wrong-import-position
from .__main__ import repeated_launch


@genty.genty
class BenchmarkTests(TestCase):

    @genty.genty_dataset(  # type: ignore
        val0=(0, False),
        val1=(1, True),
        val5=(5, False),
        val6=(6, True),
    )
    def test_moduler(self, value: int, expected: bool) -> None:
        moduler = core.Moduler(5, 1)
        np.testing.assert_equal(moduler(value), expected)


def test_compute() -> None:
    output = core.compute("basic")
    assert isinstance(output, core.tools.Selector)


def test_commandline_launch() -> None:
    with tempfile.TemporaryDirectory() as folder:
        output = Path(folder) / "benchmark_launch_test.csv"
        # commandline test
        CommandFunction(command=["python", "-m", "nevergrad.benchmark", "additional_experiment",
                                 "--cap_index", "2", "--num_workers", "2", "--output", str(output),
                                 "--imports", str(Path(__file__).parent / "additional" / "example.py")])()
        assert output.exists()
        df = core.tools.Selector.read_csv(str(output))
        testing.assert_set_equal(df.columns, DESCRIPTION_KEYS | {"offset"})  # "offset" comes from the custom function
        np.testing.assert_equal(len(df), 2)


def test_launch() -> None:
    with tempfile.TemporaryDirectory() as folder:
        with patch("nevergrad.benchmark.plotting.create_plots"):  # dont plot
            output = Path(folder) / "benchmark_launch_test.csv"
            # commandline test
            repeated_launch("repeated_basic", cap_index=4, num_workers=2, output=output, plot=True)
            assert output.exists()
            df = core.tools.Selector.read_csv(str(output))
            assert isinstance(df, core.tools.Selector)
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


def test_experiment_chunk_split() -> None:
    chunk = core.BenchmarkChunk(name="repeated_basic", seed=12, repetitions=2)
    chunks = chunk.split(2)
    chunks = [chunks[0]] + chunks[1].split(3)
    chained = [x[0] for x in itertools.chain.from_iterable(chunks)]
    # check full order (everythink only once)
    np.testing.assert_array_equal(chained, [0, 2, 4, 6, 8, 10, 12, 14, 16, 18,
                                            1, 7, 13, 19,
                                            3, 9, 15,
                                            5, 11, 17])
    testing.assert_set_equal(chained, range(20))
    assert chunks[0].id.endswith("_i0m2"), f"Wrong id {chunks[0].id}"
    assert chunks[0].id[:-4] == chunk.id[:-4], "Id prefix should be inherited"


def test_experiment_chunk_seeding() -> None:
    cap_index = 2
    chunk = core.BenchmarkChunk(name="repeated_basic", seed=12, repetitions=2, cap_index=cap_index)
    xps = [xp for _, xp in chunk]
    assert xps[0].seed != xps[cap_index].seed
    np.testing.assert_equal(len(xps), 2 * cap_index)
