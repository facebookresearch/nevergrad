# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from unittest import TestCase
from typing import List, Tuple, Any
import genty
import numpy as np
from ..common import testing
from ..instrumentation import Instrumentation
from ..instrumentation import variables as var
from . import optimizerlib
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
        self.logs.append(f"s{self._num_suggestions}")  # s for suggest
        return np.array((self._num_suggestions,))

    def _internal_tell(self, x: base.ArrayLike, value: float) -> None:
        self.logs.append(f"u{x[0]}")  # u for update


@genty.genty
class OptimizationTests(TestCase):

    @genty.genty_dataset(  # type: ignore
        w1_batch=(1, True, ['s0', 'u0', 's1', 'u1', 's2', 'u2', 's3', 'u3', 's4', 'u4']),
        w1_steady=(1, False, ['s0', 'u0', 's1', 'u1', 's2', 'u2', 's3', 'u3', 's4', 'u4']),  # no difference (normal, since worker=1)
        w3_batch=(3, True, ['s0', 's1', 's2', 'u0', 'u1', 'u2', 's3', 's4', 'u3', 'u4']),
        w3_steady=(3, False, ['s0', 's1', 's2', 'u0', 'u1', 'u2', 's3', 's4', 'u3', 'u4']),  # not really steady TODO change this behavior
        # w3_steady=(3, False, ['s0', 's1', 's2', 'u0', 's3', 'u1', 's4', 'u2', 'u3', 'u4']),  # This is what we would like
    )
    def test_batch_and_steady_optimization(self, num_workers: int, batch_mode: bool, expected: List[Tuple[str, float]]) -> None:
        # tests the suggestion (s) and update (u) patterns
        # the w3_steady is unexpected. It is designed to be efficient with a non-sequential executor, but because
        # of it, it is acting like batch mode when sequential...
        optim = LoggingOptimizer(num_workers=num_workers)
        func = CounterFunction()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optim.optimize(func, verbosity=2, batch_mode=batch_mode)
        testing.printed_assert_equal(optim.logs, expected)

    @genty.genty_dataset(  # type: ignore
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
    def test_tell_types(self, value: Any, error: bool) -> None:
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
    zeroptim.tell([0, 0], 0)
    zeroptim.tell([1, 1], 1)
    np.testing.assert_equal(zeroptim.provide_recommendation(), [0, 0])
    # check that the best value is updated if a second evaluation is not as good
    zeroptim.tell([0, 0], 10)
    zeroptim.tell([1, 1], 1)
    np.testing.assert_equal(zeroptim.provide_recommendation(), [1, 1])
    np.testing.assert_equal(zeroptim._num_suggestions, 1)


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
