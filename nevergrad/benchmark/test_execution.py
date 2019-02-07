# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from operator import add
from unittest import TestCase
from typing import List, Tuple, Dict, Any, Optional, Callable
import genty
import numpy as np
from . import execution


class Function(execution.PostponedObject):

    def __call__(self, x: int, y: int) -> float:
        return x + y

    # pylint: disable=unused-argument
    def get_postponing_delay(self, arguments: Tuple[Tuple[Any, ...], Dict[str, Any]], value: float) -> float:
        print("waiting", 5 - value)
        return 5 - value


@genty.genty
class ExecutorTest(TestCase):

    @genty.genty_dataset(  # type: ignore
        simple=(add, list(range(10))),
        delayed=(Function(), [5, 6, 7, 8, 9, 4, 3, 2, 1, 0])
    )
    def test_mocked_steady_executor(self, func: Callable, expected: List[int]) -> None:
        executor = execution.MockedSteadyExecutor()
        jobs: List[execution.MockedSteadyJob] = []
        for k in range(10):
            jobs.append(executor.submit(func, k, 0))
        results: List[float] = []
        while jobs:
            finished = [j for j in jobs if j.done()]
            np.testing.assert_(len(finished) == 1)
            results.append(finished[0].result())
            jobs.remove(finished[0])
        np.testing.assert_array_equal(results, expected)


def test_mocked_steady_executor_time() -> None:
    func = Function()
    executor = execution.MockedSteadyExecutor()
    jobs = [executor.submit(func, 0, 0)]
    np.testing.assert_equal(jobs[0].done(), True)
    jobs.append(executor.submit(func, 2, 0))
    np.testing.assert_equal(jobs[0].done(), False)  # now there is one job before
    np.testing.assert_equal(jobs[1].done(), True)
    np.testing.assert_equal(executor._time, 0)
    jobs[1].result()
    np.testing.assert_equal(executor._time, 3)
    # now making sure that jobs start from new time
    jobs.append(executor.submit(func, 4, 0))
    jobs.append(executor.submit(func, 3, 0))
    np.testing.assert_array_equal([j.release_time for j in executor.priority_queue], [4, 5, 5])
    new_finished: Optional[List[execution.MockedSteadyJob]] = None
    order: List[int] = []
    # pylint: disable=unsubscriptable-object
    while new_finished is None or new_finished:
        if new_finished is not None:
            assert len(new_finished) == 1, f'Weird list: {new_finished}'
            order.append(jobs.index(new_finished[0]))
            new_finished[0].result()
        print(jobs)
        new_finished = [j for j in jobs if j.done() and not j._is_read]
    np.testing.assert_array_equal(order, [2, 0, 3])
