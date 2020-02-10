# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from typing import Callable, Optional, Any
import numpy as np
from ..common.typetools import ArrayLike
from ..common import testing
from . import recaster
from . import optimizerlib


def test_message() -> None:
    message = recaster.Message(1, 2, blublu=3)
    np.testing.assert_equal(message.done, False)
    np.testing.assert_equal(message.args, [1, 2])
    np.testing.assert_equal(message.kwargs, {"blublu": 3})
    message.result = 3
    np.testing.assert_equal(message.done, True)
    np.testing.assert_equal(message.result, 3)


def fake_caller(func: Callable[[int], int]) -> int:
    output = 0
    for k in range(10):
        output += func(k)
    return output


@testing.parametrized(
    finished=(10, 30),
    unfinished=(2, None),  # should not hang at deletion!
)
def test_messaging_thread(num_iter: int, output: Optional[int]) -> None:
    thread = recaster.MessagingThread(fake_caller)
    num_answers = 0
    while num_answers < num_iter:
        if thread.messages and not thread.messages[0].done:
            thread.messages[0].result = 3
            num_answers += 1
        time.sleep(0.001)
    np.testing.assert_equal(thread.output, output)


def test_automatic_thread_deletion() -> None:
    thread = recaster.MessagingThread(fake_caller)
    assert thread.is_alive()


def fake_cost_function(x: ArrayLike) -> float:
    return float(np.sum(np.array(x) ** 2))


class FakeOptimizer(recaster.SequentialRecastOptimizer):

    def get_optimization_function(self) -> Callable[[Callable[..., Any]], ArrayLike]:
        # create a new instance to avoid deadlock
        return self.__class__(self.parametrization, self.budget, self.num_workers)._optim_function

    def _optim_function(self, func: Callable[..., Any]) -> ArrayLike:
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
