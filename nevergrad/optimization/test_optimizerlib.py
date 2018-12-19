# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from unittest import TestCase
from typing import Type
import genty
import numpy as np
from ..common.typetools import ArrayLike
from . import base
from .recaster import FinishedUnderlyingOptimizerWarning
from . import optimizerlib
from .optimizerlib import registry


def fitness(x: ArrayLike) -> float:
    return float(np.sum((np.array(x, copy=False) - np.array([0.5, -0.8]))**2))


def check_optimizer(optimizer_cls: Type[base.Optimizer], budget: int = 300, verify_value: bool = True) -> None:
    num_workers = 1 if optimizer_cls.recast else 2  # recast optimizer do not support num_workers > 1
    optimizer = optimizer_cls(dimension=2, budget=budget, num_workers=num_workers)
    with warnings.catch_warnings():
        # benchmark do not need to be efficient
        warnings.filterwarnings("ignore", category=base.InefficientSettingsWarning)
        # some optimizers finish early
        warnings.filterwarnings("ignore", category=FinishedUnderlyingOptimizerWarning)
        # now optimize :)
        output = optimizer.optimize(fitness)
    if verify_value:
        np.testing.assert_array_almost_equal(output, [0.5, -0.8], decimal=1)
    # make sure we are correctly tracking the best values
    archive = optimizer.archive
    assert (optimizer.current_bests["pessimistic"].pessimistic_confidence_bound ==
            min(v.pessimistic_confidence_bound for v in archive.values()))


SLOW = ["NoisyDE", "NoisyBandit"]


@genty.genty
class OptimizerTests(TestCase):

    @genty.genty_dataset(**{name: (name, optimizer,) for name, optimizer in registry.items() if "BO" not in name})  # type: ignore
    def test_optimizers(self, name: str, optimizer_cls: Type[base.Optimizer]) -> None:
        verify = not optimizer_cls.one_shot and name not in SLOW and "Discrete" not in name
        check_optimizer(optimizer_cls, budget=300, verify_value=verify)


def test_pso_to_real() -> None:
    output = optimizerlib.PSO.to_real([.3, .5, .9])
    np.testing.assert_almost_equal(output, [-.52, 0, 1.28], decimal=2)
    np.testing.assert_raises(AssertionError, optimizerlib.PSO.to_real, [.3, .5, 1.2])
