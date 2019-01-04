# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import warnings
from pathlib import Path
from unittest import SkipTest
from unittest import TestCase
from typing import Type
import genty
import numpy as np
import pandas as pd
from ..common.typetools import ArrayLike
from . import base
from .recaster import FinishedUnderlyingOptimizerWarning
from . import optimizerlib
from .optimizerlib import registry


def fitness(x: ArrayLike) -> float:
    """Simple quadratic fitness function which can be used with dimension up to 4
    """
    x0 = [0.5, -0.8, 0, 4][:len(x)]
    return float(np.sum((np.array(x, copy=False) - x0)**2))


def check_optimizer(optimizer_cls: Type[base.Optimizer], budget: int = 300, verify_value: bool = True) -> None:
    # recast optimizer do not support num_workers > 1, and respect no_parallelization.
    num_workers = (1 if optimizer_cls.recast or optimizer_cls.no_parallelization else 2)
    optimizer = optimizer_cls(dimension=2, budget=budget, num_workers=num_workers)
    num_attempts = 1 if not verify_value else 2  # allow 2 attemps to get to the optimum (shit happens...)
    for k in range(1, num_attempts + 1):
        with warnings.catch_warnings():
            # benchmark do not need to be efficient
            warnings.filterwarnings("ignore", category=base.InefficientSettingsWarning)
            # some optimizers finish early
            warnings.filterwarnings("ignore", category=FinishedUnderlyingOptimizerWarning)
            # now optimize :)
            output = optimizer.optimize(fitness)
        if verify_value:
            try:
                np.testing.assert_array_almost_equal(output, [0.5, -0.8], decimal=1)
            except AssertionError as e:
                print(f"Attemp #{k}: failed with value {tuple(output)}")
                if k == num_attempts:
                    raise e
            else:
                break
    # make sure we are correctly tracking the best values
    archive = optimizer.archive
    assert (optimizer.current_bests["pessimistic"].pessimistic_confidence_bound ==
            min(v.pessimistic_confidence_bound for v in archive.values()))


SLOW = ["NoisyDE", "NoisyBandit", "SPSA"]


@genty.genty
class OptimizerTests(TestCase):

    recommendations = pd.DataFrame(columns=[f"v{k}" for k in range(4)])
    _RECOM_FILE = Path(__file__).parent / "recorded_recommendations.csv"

    @classmethod
    def setUpClass(cls) -> None:
        # load recorded recommendations
        if cls._RECOM_FILE.exists():
            cls.recommendations = pd.read_csv(cls._RECOM_FILE, index_col=0)

    @classmethod
    def tearDownClass(cls) -> None:
        # sort and remove unused names
        # then update recommendation file
        names = sorted(x for x in cls.recommendations.index if x in registry)
        recom = cls.recommendations.loc[names, :]
        recom.iloc[:, 1:] = np.round(recom.iloc[:, 1:], 12)
        recom.to_csv(cls._RECOM_FILE)

    @genty.genty_dataset(**{name: (name, optimizer,) for name, optimizer in registry.items() if "BO" not in name})  # type: ignore
    def test_optimizers(self, name: str, optimizer_cls: Type[base.Optimizer]) -> None:
        verify = not optimizer_cls.one_shot and name not in SLOW and "Discrete" not in name
        check_optimizer(optimizer_cls, budget=300, verify_value=verify)

    @genty.genty_dataset(**{name: (name, optimizer,) for name, optimizer in registry.items() if "BO" not in name})  # type: ignore
    def test_optimizers_recommendation(self, name: str, optimizer_cls: Type[base.Optimizer]) -> None:
        if "CMA" in name:
            raise SkipTest("Not playing nicely with the tests")  # thread problem?
        np.random.seed(12)
        if optimizer_cls.recast:
            random.seed(12)  # may depend on non numpy generator
        optim = optimizer_cls(dimension=4, budget=6, num_workers=1)
        output = optim.optimize(fitness)
        if name not in self.recommendations.index:
            self.recommendations.loc[name, :] = tuple(output)
            raise ValueError(f'Recorded the value for optimizer "{name}", please rerun this test locally.')
        np.testing.assert_array_almost_equal(output, self.recommendations.loc[name, :], decimal=10,
                                             err_msg="Something has changed, if this is normal, delete the following "
                                             f"file and rerun to update the values:\n{self._RECOM_FILE}")


def test_pso_to_real() -> None:
    output = optimizerlib.PSO.to_real([.3, .5, .9])
    np.testing.assert_almost_equal(output, [-.52, 0, 1.28], decimal=2)
    np.testing.assert_raises(AssertionError, optimizerlib.PSO.to_real, [.3, .5, 1.2])
