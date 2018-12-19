# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import contextlib
from unittest import TestCase
from unittest.mock import patch
from typing import Optional, List
import genty
import numpy as np
from ..common import testing
from ..functions import ArtificialFunction
from ..functions.test_functionlib import DESCRIPTION_KEYS as ARTIFICIAL_KEYS
from .xpbase import Experiment
from .xpbase import create_seed_generator


DESCRIPTION_KEYS = {"seed", "elapsed_time", "elapsed_budget", "loss", "optimizer_name", "num_workers", "budget", "error"} | ARTIFICIAL_KEYS


def test_run_artificial_function() -> None:
    func = ArtificialFunction(name="sphere", block_dimension=2)
    xp = Experiment(func, optimizer_name="OnePlusOne", budget=300, num_workers=1)
    summary = xp.run()
    assert summary["elapsed_time"] < .5  # should be much faster
    assert summary["loss"] < .001
    testing.assert_set_equal(summary.keys(), DESCRIPTION_KEYS)
    np.testing.assert_equal(summary["elapsed_budget"], 300)


def test_run_with_error() -> None:
    func = ArtificialFunction(name="sphere", block_dimension=2)
    xp = Experiment(func, optimizer_name="OnePlusOne", budget=300, num_workers=1)
    with patch("nevergrad.benchmark.xpbase.Experiment._run_with_error") as run:
        run.side_effect = ValueError("test error string")
        with contextlib.redirect_stderr(sys.stdout):
            summary = xp.run()
    testing.assert_set_equal(summary.keys(), DESCRIPTION_KEYS)
    np.testing.assert_equal(summary["error"], "ValueError")


@genty.genty
class ExperimentsTests(TestCase):

    @genty.genty_dataset(  # type: ignore
        concurrent=("OnePlusOne", 10, False),  # no true case implemented for now
    )
    def test_is_incoherent(self, optimizer: str, num_workers: int, expected: bool) -> None:
        func = ArtificialFunction(name="sphere", block_dimension=2)
        xp = Experiment(func, optimizer_name=optimizer, budget=300, num_workers=num_workers)
        np.testing.assert_equal(xp.is_incoherent, expected)

    @genty.genty_dataset(  # type: ignore
        none=(None, 12, [None, None, None, None]),
        seed_no_rand=(12, 0, [363, 803, 222, 277]),
        seed_with_rand=(12, 12, [363, 803, 222, 277]),
        different_seed=(24, 0, [914, 555, 376, 855]),
    )
    def test_seed_generator(self, seed: Optional[int], randsize: int, expected: List[Optional[int]]) -> None:
        output = []
        generator = create_seed_generator(seed)
        for _ in range(4):
            if randsize:  # call the standard random generator
                np.random.normal(0, 1, size=randsize)
            value = next(generator)
            output.append(value if value is None else value % 1000)
        np.testing.assert_array_equal(output, expected)


def test_equality() -> None:
    func = ArtificialFunction(name="sphere", block_dimension=2)
    xp1 = Experiment(func, optimizer_name="OnePlusOne", budget=300, num_workers=2)
    xp2 = Experiment(func, optimizer_name="RandomSearch", budget=300, num_workers=2)
    assert xp1 != xp2
