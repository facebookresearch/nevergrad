# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import warnings
from pathlib import Path
from unittest import SkipTest
from typing import Type, Union, Generator
import pytest
import numpy as np
import pandas as pd
from ..common.typetools import ArrayLike
from ..common import testing
from . import base
from .recaster import FinishedUnderlyingOptimizerWarning
from . import optimizerlib
from .optimizerlib import registry


def fitness(x: ArrayLike) -> float:
    """Simple quadratic fitness function which can be used with dimension up to 4
    """
    x0 = [0.5, -0.8, 0, 4][:len(x)]
    return float(np.sum((np.array(x, copy=False) - x0)**2))


def check_optimizer(optimizer_cls: Union[base.OptimizerFamily, Type[base.Optimizer]], budget: int = 300, verify_value: bool = True) -> None:
    # recast optimizer do not support num_workers > 1, and respect no_parallelization.
    num_workers = (1 if optimizer_cls.recast or optimizer_cls.no_parallelization else 2)
    num_attempts = 1 if not verify_value else 2  # allow 2 attemps to get to the optimum (shit happens...)
    for k in range(1, num_attempts + 1):
        optimizer = optimizer_cls(dimension=2, budget=budget, num_workers=num_workers)
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
    # add a random point to test tell_not_asked
    try:
        optimizer.tell_not_asked(np.random.normal(0, 1, size=optimizer.dimension), 12.)
    except Exception as e:  # pylint: disable=broad-except
        if not isinstance(e, base.TellNotAskedNotSupportedError):
            raise AssertionError("Optimizers should raise base.TellNotAskedNotSupportedError "
                                 "at tell_not_asked if they do not support it") from e


SLOW = ["NoisyDE", "NoisyBandit", "SPSA", "NoisyOnePlusOne", "OptimisticNoisyOnePlusOne", "ASCMADEthird", "ASCMA2PDEthird", "MultiScaleCMA",
        "PCEDA", "MPCEDA", "EDA", "MEDA", "MicroCMA"]
UNSEEDABLE = ["CMA", "Portfolio", "ASCMADEthird", "ASCMADEQRthird", "ASCMA2PDEthird", "CMandAS2", "DiagonalCMA",
              "CMandAS", "CM", "MultiCMA", "TripleCMA", "MultiScaleCMA", "MilliCMA", "MicroCMA"]


@testing.parametrized(**{name: (name, optimizer,) for name, optimizer in registry.items()})
def test_optimizers(name: str, optimizer_cls: Union[base.OptimizerFamily, Type[base.Optimizer]]) -> None:
    if isinstance(optimizer_cls, base.OptimizerFamily):
        assert hasattr(optimizerlib, name)  # make sure registration matches name in optimizerlib
    verify = not optimizer_cls.one_shot and name not in SLOW and not any(x in name for x in ["BO", "Discrete"])
    # BO is extremely slow, run it anyway but very low budget and no verification
    check_optimizer(optimizer_cls, budget=2 if "BO" in name else 300, verify_value=verify)


class RecommendationKeeper:

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath
        self.recommendations = pd.DataFrame(columns=[f"v{k}" for k in range(4)])
        if filepath.exists():
            self.recommendations = pd.read_csv(filepath, index_col=0)

    def save(self) -> None:
        # sort and remove unused names
        # then update recommendation file
        names = sorted(x for x in self.recommendations.index if x in registry)
        recom = self.recommendations.loc[names, :]
        recom.iloc[:, 1:] = np.round(recom.iloc[:, 1:], 12)
        recom.to_csv(self.filepath)


@pytest.fixture(scope="module")  # type: ignore
def recomkeeper() -> Generator[RecommendationKeeper, None, None]:
    keeper = RecommendationKeeper(filepath=Path(__file__).parent / "recorded_recommendations.csv")
    yield keeper
    keeper.save()


@pytest.mark.parametrize("name", [name for name in registry if "BO" not in name])  # type: ignore
def test_optimizers_recommendation(name: str, recomkeeper: RecommendationKeeper) -> None:  # pylint: disable=redefined-outer-name
    optimizer_cls = registry[name]
    if name in UNSEEDABLE:
        raise SkipTest("Not playing nicely with the tests (unseedable)")  # due to CMA not seedable.
    np.random.seed(12)
    if optimizer_cls.recast:
        random.seed(12)  # may depend on non numpy generator
    optim = optimizer_cls(dimension=4, budget=6, num_workers=1)
    np.testing.assert_equal(optim.name, name)
    output = optim.optimize(fitness)
    if name not in recomkeeper.recommendations.index:
        recomkeeper.recommendations.loc[name, :] = tuple(output)
        raise ValueError(f'Recorded the value for optimizer "{name}", please rerun this test locally.')
    np.testing.assert_array_almost_equal(output, recomkeeper.recommendations.loc[name, :], decimal=10,
                                         err_msg="Something has changed, if this is normal, delete the following "
                                         f"file and rerun to update the values:\n{recomkeeper.filepath}")


@testing.parametrized(
    de=("DE", 10, 10, 30),
    de_w=("DE", 50, 40, 40),
    de1=("OnePointDE", 10, 10, 30),
    de1_w=("OnePointDE", 50, 40, 40),
    dim_d=("AlmostRotationInvariantDEAndBigPop", 50, 40, 51),
    dim=("AlmostRotationInvariantDEAndBigPop", 10, 40, 40),
    dim_d_rot=("RotationInvariantDE", 50, 40, 51),
    large=("BPRotationInvariantDE", 10, 40, 70),
)
def test_differential_evolution_popsize(name: str, dimension: int, num_workers: int, expected: int) -> None:
    optim = registry[name](dimension=dimension, budget=100, num_workers=num_workers)
    np.testing.assert_equal(optim.llambda, expected)


def test_pso_to_real() -> None:
    output = optimizerlib.PSO.to_real([.3, .5, .9])
    np.testing.assert_almost_equal(output, [-.52, 0, 1.28], decimal=2)
    np.testing.assert_raises(AssertionError, optimizerlib.PSO.to_real, [.3, .5, 1.2])


def test_portfolio_budget() -> None:
    for k in range(3, 13):
        optimizer = optimizerlib.Portfolio(dimension=2, budget=k)
        np.testing.assert_equal(optimizer.budget, sum(o.budget for o in optimizer.optims))


def test_optimizer_families_repr() -> None:
    Cls = optimizerlib.DifferentialEvolution
    np.testing.assert_equal(repr(Cls()), "DifferentialEvolution()")
    np.testing.assert_equal(repr(Cls(initialization='LHS')), "DifferentialEvolution(initialization='LHS')")
    #
    optimrs = optimizerlib.RandomSearchMaker(cauchy=True)
    np.testing.assert_equal(repr(optimrs), "RandomSearchMaker(cauchy=True)")
    #
    optimso = optimizerlib.ScipyOptimizer(method="COBYLA")
    np.testing.assert_equal(repr(optimso), "ScipyOptimizer(method='COBYLA')")
    assert optimso.no_parallelization
