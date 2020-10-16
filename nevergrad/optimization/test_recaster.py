# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import numpy as np
import nevergrad as ng
import nevergrad.common.typing as tp
from nevergrad.common import testing
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


def fake_caller(func: tp.Callable[[int], int]) -> int:
    output = 0
    for k in range(10):
        output += func(k)
    return output


@testing.parametrized(
    finished=(10, 30),
    unfinished=(2, None),  # should not hang at deletion!
)
def test_messaging_thread(num_iter: int, output: tp.Optional[int]) -> None:
    thread = recaster.MessagingThread(fake_caller)
    num_answers = 0
    while num_answers < num_iter:
        if thread.messages and not thread.messages[0].done:
            thread.messages[0].result = 3
            num_answers += 1
        time.sleep(0.001)
    with testing.skip_error_on_systems(AssertionError, systems=("Windows",)):  # TODO fix
        np.testing.assert_equal(thread.output, output)


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
    assert isinstance(opt.provide_recommendation(), ng.p.Parameter), "Recommendation should be available from start"
    # the recommended solution should be the better one among the told points
    x1 = opt.ask()
    opt.tell(x1, 10)
    x2 = opt.ask()
    opt.tell(x2, 5)
    recommendation = opt.provide_recommendation()
    np.testing.assert_array_almost_equal(recommendation.value, x2.value)
