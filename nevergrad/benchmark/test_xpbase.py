# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import contextlib
from unittest.mock import patch
import numpy as np
import nevergrad.common.typing as tp
from nevergrad.common import testing
from nevergrad.optimization import test_base
from nevergrad.functions import ArtificialFunction
from nevergrad.functions.base import MultiExperiment
from nevergrad.functions.test_base import ExampleFunction
from nevergrad.functions.test_functionlib import DESCRIPTION_KEYS as ARTIFICIAL_KEYS
from . import xpbase


DESCRIPTION_KEYS = {
    "seed",
    "elapsed_time",
    "elapsed_budget",
    "loss",
    "optimizer_name",
    "pseudotime",
    "num_workers",
    "budget",
    "error",
    "batch_mode",
} | ARTIFICIAL_KEYS


def test_run_artificial_function() -> None:
    func = ArtificialFunction(name="sphere", block_dimension=2)
    xp = xpbase.Experiment(func, optimizer="OnePlusOne", budget=24, num_workers=2, batch_mode=True, seed=12)
    summary = xp.run()
    assert summary["elapsed_time"] < 0.5  # should be much faster
    np.testing.assert_almost_equal(summary["loss"], 0.08444784112287358)  # makes sure seeding works!
    testing.assert_set_equal(summary.keys(), DESCRIPTION_KEYS)
    np.testing.assert_equal(summary["elapsed_budget"], 24)
    np.testing.assert_equal(summary["pseudotime"], 12)  # defaults to 1 unit per eval ( /2 because 2 workers)


def test_run_packed_artificial_function() -> None:
    func = MultiExperiment(
        [ArtificialFunction(name="sphere", block_dimension=2) for _ in range(2)], [100, 100]
    )
    xp = xpbase.Experiment(func, optimizer="OnePlusOne", budget=24, num_workers=2, batch_mode=True, seed=14)
    summary = xp.run()
    np.testing.assert_almost_equal(summary["loss"], -9784.8, decimal=1)  # makes sure seeding works!


def test_noisy_artificial_function_loss() -> None:
    func = ArtificialFunction(name="sphere", block_dimension=5, noise_level=0.3)
    seed = np.random.randint(99999)
    xp = xpbase.Experiment(func, optimizer="OnePlusOne", budget=5, seed=seed)
    # Because copy() can have different random initialization for the parameters
    # The function should be copied early.
    np.random.seed(seed)
    pfunc = func.copy()
    xp.run()
    loss_ref = xp.result["loss"]
    # now with copy
    assert xp._optimizer is not None
    reco = xp._optimizer.provide_recommendation()
    assert reco is not None
    np.testing.assert_equal(pfunc.evaluation_function(reco), loss_ref)
    np.random.seed(None)


def test_run_with_error() -> None:
    func = ArtificialFunction(name="sphere", block_dimension=2)
    xp = xpbase.Experiment(func, optimizer="OnePlusOne", budget=300, num_workers=1)
    with patch("nevergrad.optimization.base.Optimizer.minimize") as run:
        run.side_effect = ValueError("test error string")
        with contextlib.redirect_stderr(sys.stdout):
            summary = xp.run()
    testing.assert_set_equal(summary.keys(), DESCRIPTION_KEYS)
    np.testing.assert_equal(summary["error"], "ValueError")
    assert xp._optimizer is not None
    np.testing.assert_equal(
        xp._optimizer.num_tell, 0
    )  # make sure optimizer is kept in case we need to restart (eg.: KeyboardInterrupt)
    assert not np.isnan(summary["loss"]), "Loss should be recorded with the current recommendation"


@testing.parametrized(
    concurrent=("OnePlusOne", 10, False),  # no true case implemented for now
)
def test_is_incoherent(optimizer: str, num_workers: int, expected: bool) -> None:
    func = ArtificialFunction(name="sphere", block_dimension=2)
    xp = xpbase.Experiment(func, optimizer=optimizer, budget=300, num_workers=num_workers)
    np.testing.assert_equal(xp.is_incoherent, expected)


@testing.parametrized(
    none=(None, 12, [None, None, None, None]),
    seed_no_rand=(12, 0, [363, 803, 222, 277]),
    seed_with_rand=(12, 12, [363, 803, 222, 277]),
    different_seed=(24, 0, [914, 555, 376, 855]),
)
def test_seed_generator(seed: tp.Optional[int], randsize: int, expected: tp.List[tp.Optional[int]]) -> None:
    output = []
    generator = xpbase.create_seed_generator(seed)
    for _ in range(4):
        if randsize:  # call the standard random generator
            np.random.normal(0, 1, size=randsize)
        value = next(generator)
        output.append(value if value is None else value % 1000)
    np.testing.assert_array_equal(output, expected)


@testing.parametrized(
    w3_batch=(True, ["s0", "s1", "s2", "u0", "u1", "u2", "s3", "s4", "u3", "u4"]),
    w3_steady=(False, ["s0", "s1", "s2", "u2", "s3", "u1", "s4", "u0", "u3", "u4"]),  # u0 and u1 are delayed
)
def test_batch_mode_parameter(batch_mode: bool, expected: tp.List[str]) -> None:
    func = ExampleFunction(dimension=1, number=3)
    optim = test_base.LoggingOptimizer(3)
    with patch.object(xpbase.OptimizerSettings, "instantiate", return_value=optim):
        xp = xpbase.Experiment(func, optimizer="OnePlusOne", budget=10, num_workers=3, batch_mode=batch_mode)
        xp._run_with_error()
        testing.printed_assert_equal(optim.logs, expected)


def test_equality() -> None:
    func = ArtificialFunction(name="sphere", block_dimension=2)
    xp1 = xpbase.Experiment(func, optimizer="OnePlusOne", budget=300, num_workers=2)
    xp2 = xpbase.Experiment(func, optimizer="RandomSearch", budget=300, num_workers=2)
    assert xp1 != xp2


def test_multiobjective_experiment() -> None:
    mofunc = MultiExperiment(
        [ArtificialFunction("sphere", block_dimension=7), ArtificialFunction("cigar", block_dimension=7)],
        upper_bounds=np.array((50.0, 50.0)),
    )
    xp = xpbase.Experiment(mofunc, optimizer="TwoPointsDE", budget=100, num_workers=1)
    summary = xp.run()
    loss: float = summary["loss"]
    assert loss < 1e9
