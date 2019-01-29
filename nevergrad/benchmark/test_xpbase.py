# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import contextlib
from unittest import TestCase
from unittest.mock import patch
from typing import Optional, List, Tuple, Any, Dict
import genty
import numpy as np
from ..common import testing
from ..optimization import test_base
from ..functions.base import BaseFunction
from ..functions import ArtificialFunction
from ..functions.test_functionlib import DESCRIPTION_KEYS as ARTIFICIAL_KEYS
from . import execution
from . import xpbase


DESCRIPTION_KEYS = {"seed", "elapsed_time", "elapsed_budget", "loss", "optimizer_name",
                    "num_workers", "budget", "error", "batch_mode"} | ARTIFICIAL_KEYS


class Function(BaseFunction, execution.PostponedObject):

    def oracle_call(self, x: np.ndarray) -> float:
        return float(x[0])

    # pylint: disable=unused-argument
    def get_postponing_delay(self, arguments: Tuple[Tuple[Any, ...], Dict[str, Any]], value: float) -> float:
        return 5 - value


def test_run_artificial_function() -> None:
    func = ArtificialFunction(name="sphere", block_dimension=2)
    xp = xpbase.Experiment(func, optimizer_name="OnePlusOne", budget=300, num_workers=1)
    summary = xp.run()
    assert summary["elapsed_time"] < .5  # should be much faster
    assert summary["loss"] < .001
    testing.assert_set_equal(summary.keys(), DESCRIPTION_KEYS)
    np.testing.assert_equal(summary["elapsed_budget"], 300)


def test_run_with_error() -> None:
    func = ArtificialFunction(name="sphere", block_dimension=2)
    xp = xpbase.Experiment(func, optimizer_name="OnePlusOne", budget=300, num_workers=1)
    with patch("nevergrad.optimization.optimizerlib.OnePlusOne.optimize") as run:
        run.side_effect = ValueError("test error string")
        with contextlib.redirect_stderr(sys.stdout):
            summary = xp.run()
    testing.assert_set_equal(summary.keys(), DESCRIPTION_KEYS)
    np.testing.assert_equal(summary["error"], "ValueError")
    assert not np.isnan(summary["loss"]), "Loss should be recorded with the current recommendation"


@genty.genty
class ExperimentsTests(TestCase):

    @genty.genty_dataset(  # type: ignore
        concurrent=("OnePlusOne", 10, False),  # no true case implemented for now
    )
    def test_is_incoherent(self, optimizer: str, num_workers: int, expected: bool) -> None:
        func = ArtificialFunction(name="sphere", block_dimension=2)
        xp = xpbase.Experiment(func, optimizer_name=optimizer, budget=300, num_workers=num_workers)
        np.testing.assert_equal(xp.is_incoherent, expected)

    @genty.genty_dataset(  # type: ignore
        none=(None, 12, [None, None, None, None]),
        seed_no_rand=(12, 0, [363, 803, 222, 277]),
        seed_with_rand=(12, 12, [363, 803, 222, 277]),
        different_seed=(24, 0, [914, 555, 376, 855]),
    )
    def test_seed_generator(self, seed: Optional[int], randsize: int, expected: List[Optional[int]]) -> None:
        output = []
        generator = xpbase.create_seed_generator(seed)
        for _ in range(4):
            if randsize:  # call the standard random generator
                np.random.normal(0, 1, size=randsize)
            value = next(generator)
            output.append(value if value is None else value % 1000)
        np.testing.assert_array_equal(output, expected)

    @genty.genty_dataset(  # type: ignore
        w3_batch=(True, ['s0', 's1', 's2', 'u0', 'u1', 'u2', 's3', 's4', 'u3', 'u4']),
        w3_steady=(False, ['s0', 's1', 's2', 'u2', 's3', 'u1', 's4', 'u0', 'u3', 'u4']),  # u0 and u1 are delayed
    )
    def test_batch_mode_parameter(self, batch_mode: bool, expected: List[str]) -> None:
        func = Function(dimension=1)
        optim = test_base.LoggingOptimizer(3)
        with patch.object(xpbase.OptimizerSettings, "instanciate", return_value=optim):
            xp = xpbase.Experiment(func, optimizer_name="OnePlusOne", budget=10, num_workers=3, batch_mode=batch_mode)
            xp._run_with_error()
            testing.printed_assert_equal(optim.logs, expected)


def test_equality() -> None:
    func = ArtificialFunction(name="sphere", block_dimension=2)
    xp1 = xpbase.Experiment(func, optimizer_name="OnePlusOne", budget=300, num_workers=2)
    xp2 = xpbase.Experiment(func, optimizer_name="RandomSearch", budget=300, num_workers=2)
    assert xp1 != xp2
