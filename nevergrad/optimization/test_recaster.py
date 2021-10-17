# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import nevergrad as ng
import nevergrad.common.typing as tp
from nevergrad.functions import ArtificialFunction
from nevergrad.benchmark import experiments
from nevergrad.benchmark.xpbase import Experiment
from . import recaster
from . import optimizerlib


def fake_caller(func: tp.Callable[[int], int]) -> int:
    output = 0
    for k in range(10):
        output += func(k)
    return output


def test_automatic_thread_deletion() -> None:
    thread = recaster.MessagingThread(fake_caller)
    assert thread.is_alive()


def fake_cost_function(x: tp.ArrayLike) -> float:
    return float(np.sum(np.array(x) ** 2))


class FakeOptimizer(recaster.SequentialRecastOptimizer):
    def get_optimization_function(self) -> tp.Callable[[tp.Callable[..., tp.Any]], tp.ArrayLike]:
        # create a new instance to avoid deadlock
        return self.__class__(self.parametrization, self.budget, self.num_workers)._optim_function

    def _optim_function(self, func: tp.Callable[..., tp.Any]) -> tp.ArrayLike:
        suboptim = optimizerlib.OnePlusOne(parametrization=2, budget=self.budget)
        recom = suboptim.minimize(func)
        return recom.get_standardized_data(reference=self.parametrization)


def test_recast_optimizer() -> None:
    optimizer = FakeOptimizer(parametrization=2, budget=100)
    optimizer.minimize(fake_cost_function)
    assert optimizer._messaging_thread is not None
    np.testing.assert_equal(optimizer._messaging_thread._thread.call_count, 100)


def test_recast_optimizer_with_error() -> None:
    optimizer = FakeOptimizer(parametrization=2, budget=100)
    np.testing.assert_raises(TypeError, optimizer.minimize)  # did hang in some versions


def test_recast_optimizer_and_stop() -> None:
    optimizer = FakeOptimizer(parametrization=2, budget=100)
    optimizer.ask()
    # thread is not finished... but should not hang!


def test_provide_recommendation() -> None:
    opt = optimizerlib.SQP(parametrization=2, budget=100)
    assert isinstance(
        opt.provide_recommendation(), ng.p.Parameter
    ), "Recommendation should be available from start"
    # the recommended solution should be the better one among the told points
    x1 = opt.ask()
    opt.tell(x1, 10)
    x2 = opt.ask()
    opt.tell(x2, 5)
    recommendation = opt.provide_recommendation()
    np.testing.assert_array_almost_equal(recommendation.value, x2.value)


def test_sqp_with_constraint() -> None:
    func = ArtificialFunction("ellipsoid", block_dimension=10, rotation=True, translation_factor=0.1)
    func.parametrization.register_cheap_constraint(experiments._Constraint("sum", as_bool=True))
    xp = Experiment(func, optimizer="ChainMetaModelSQP", budget=150, seed=4290846341)
    xp._run_with_error()


class ErroringSequentialRecastOptimizer(recaster.SequentialRecastOptimizer):
    def get_optimization_function(self) -> tp.Callable[[tp.Callable[..., tp.Any]], tp.ArrayLike]:
        def optim_function(func: tp.Callable[..., tp.Any]) -> tp.ArrayLike:
            func(np.zeros(2))
            raise ValueError("ErroringOptimizer")

        return optim_function


class ErroringBatchRecastOptimizer(recaster.BatchRecastOptimizer):

    # pylint: disable=abstract-method

    def get_optimization_function(self) -> tp.Callable[[tp.Callable[..., tp.Any]], tp.ArrayLike]:
        def optim_function(func: tp.Callable[..., tp.Any]) -> tp.ArrayLike:
            func(np.zeros((1, 2)))
            raise ValueError("ErroringOptimizer")

        return optim_function


def test_recast_optimizer_error() -> None:
    sequential = ErroringSequentialRecastOptimizer(parametrization=2, budget=100)
    sequential.tell(sequential.ask(), 4.2)
    with pytest.raises(ValueError, match="ErroringOptimizer"):
        sequential.ask()

    batch = ErroringBatchRecastOptimizer(parametrization=2, budget=100)
    batch.tell(batch.ask(), 4.2)
    with pytest.raises(ValueError, match="ErroringOptimizer"):
        batch.ask()
