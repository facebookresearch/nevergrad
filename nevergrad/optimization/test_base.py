# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import List, Tuple, Any, Optional
import numpy as np
from ..common import testing
from ..instrumentation import Instrumentation
from ..instrumentation import variables as var
from . import optimizerlib
from . import test_optimizerlib
from . import base


class CounterFunction:

    def __init__(self) -> None:
        self.count = 0

    def __call__(self, value: base.ArrayLike) -> float:
        assert len(value) == 1
        self.count += 1
        return float(value[0] - 1)**2


class LoggingOptimizer(base.Optimizer):

    def __init__(self, num_workers: int = 1) -> None:
        super().__init__(dimension=1, budget=5, num_workers=num_workers)
        self.logs: List[str] = []

    def _internal_ask(self) -> base.ArrayLike:
        self.logs.append(f"s{self._num_ask}")  # s for suggest
        return np.array((float(self._num_ask),))

    def _internal_tell(self, x: base.ArrayLike, value: float) -> None:
        self.logs.append(f"u{int(x[0])}")  # u for update


@testing.parametrized(
    w1_batch=(1, True, ['s0', 'u0', 's1', 'u1', 's2', 'u2', 's3', 'u3', 's4', 'u4']),
    w1_steady=(1, False, ['s0', 'u0', 's1', 'u1', 's2', 'u2', 's3', 'u3', 's4', 'u4']),  # no difference (normal, since worker=1)
    w3_batch=(3, True, ['s0', 's1', 's2', 'u0', 'u1', 'u2', 's3', 's4', 'u3', 'u4']),
    w3_steady=(3, False, ['s0', 's1', 's2', 'u0', 'u1', 'u2', 's3', 's4', 'u3', 'u4']),  # not really steady TODO change this behavior
    # w3_steady=(3, False, ['s0', 's1', 's2', 'u0', 's3', 'u1', 's4', 'u2', 'u3', 'u4']),  # This is what we would like
)
def test_batch_and_steady_optimization(num_workers: int, batch_mode: bool, expected: List[Tuple[str, float]]) -> None:
    # tests the suggestion (s) and update (u) patterns
    # the w3_steady is unexpected. It is designed to be efficient with a non-sequential executor, but because
    # of it, it is acting like batch mode when sequential...
    optim = LoggingOptimizer(num_workers=num_workers)
    func = CounterFunction()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optim.optimize(func, verbosity=2, batch_mode=batch_mode)
    testing.printed_assert_equal(optim.logs, expected)


@testing.parametrized(
    int_val=(3, False),
    bool_val=(True, False),
    int32_val=(np.int32(3), False),
    int64_val=(np.int64(3), False),
    float64_val=(np.float64(3), False),
    float32_val=(np.float32(3), False),
    list_val=([3, 5], True),
    complex_val=(1j, True),
    object_val=(object(), True),
)
def test_tell_types(value: Any, error: bool) -> None:
    optim = LoggingOptimizer(num_workers=1)
    x = optim.ask()
    if error:
        np.testing.assert_raises(TypeError, optim.tell, x, value)
    else:
        optim.tell(x, value)


def test_base_optimizer() -> None:
    zeroptim = optimizerlib.Zero(dimension=2, budget=4, num_workers=1)
    representation = repr(zeroptim)
    assert "dimension=2" in representation, f"Unexpected representation: {representation}"
    np.testing.assert_equal(zeroptim.ask(), [0, 0])
    zeroptim.tell([0., 0], 0)
    zeroptim.tell([1., 1], 1)
    np.testing.assert_equal(zeroptim.provide_recommendation(), [0, 0])
    # check that the best value is updated if a second evaluation is not as good
    zeroptim.tell([0., 0], 10)
    zeroptim.tell([1., 1], 1)
    np.testing.assert_equal(zeroptim.provide_recommendation(), [1, 1])
    np.testing.assert_equal(zeroptim._num_ask, 1)


def test_optimize() -> None:
    optimizer = optimizerlib.OnePlusOne(dimension=1, budget=100, num_workers=5)
    optimizer.register_callback("tell", base.OptimizationPrinter(num_eval=10, num_sec=.1))
    func = CounterFunction()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = optimizer.optimize(func, verbosity=2)
    np.testing.assert_almost_equal(result[0], 1, decimal=2)
    np.testing.assert_equal(func.count, 100)


def test_instrumented_optimizer() -> None:
    instru = Instrumentation(var.Gaussian(0, 1), var.SoftmaxCategorical([0, 1]))
    opt = optimizerlib.registry["RandomSearch"](dimension=instru.dimension, budget=10)
    iopt = base.IntrumentedOptimizer(opt, instru)
    output = iopt.optimize(lambda x, y: x**2 + y)  # type: ignore
    assert isinstance(output, base.ArgPoint)


class StupidFamily(base.OptimizerFamily):

    def __call__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> base.Optimizer:
        class_ = base.registry["Zero"] if self._kwargs.get("zero", True) else base.registry["StupidRandom"]
        run = class_(dimension=dimension, budget=budget, num_workers=num_workers)
        run.name = self._repr
        return run  # type: ignore


def test_optimizer_family() -> None:
    for zero in [True, False]:
        optf = StupidFamily(zero=zero)
        opt = optf(dimension=2, budget=4, num_workers=1)
        recom = opt.optimize(test_optimizerlib.Fitness([.5, -.8]))
        np.testing.assert_equal(recom == np.zeros(2), zero)


def test_naming() -> None:
    optf = StupidFamily(zero=True)
    opt = optf(dimension=2, budget=4, num_workers=1)
    np.testing.assert_equal(repr(opt), "Instance of StupidFamily(zero=True)(dimension=2, budget=4, num_workers=1)")
    optf.with_name("BlubluOptimizer", register=True)
    opt = base.registry["BlubluOptimizer"](dimension=2, budget=4, num_workers=1)
    np.testing.assert_equal(repr(opt), "Instance of BlubluOptimizer(dimension=2, budget=4, num_workers=1)")
