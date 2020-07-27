# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from operator import add
import typing as tp
import numpy as np
import nevergrad as ng
from nevergrad.common import testing
from nevergrad.functions import ArtificialFunction
from nevergrad.functions import ExperimentFunction
from . import execution


class Function(ExperimentFunction):

    def __init__(self) -> None:
        super().__init__(self._func, ng.p.Instrumentation())

    def _func(self, x: int, y: int) -> float:
        return x + y

    # pylint: disable=unused-argument
    def compute_pseudotime(self, input_parameter: tp.Any, value: float) -> float:
        return 5 - value


@testing.parametrized(
    simple=(add, list(range(10))),
    delayed=(Function(), [5, 6, 7, 8, 9, 4, 3, 2, 1, 0])
)
def test_mocked_steady_executor(func: tp.Callable[..., tp.Any], expected: tp.List[int]) -> None:
    executor = execution.MockedTimedExecutor(batch_mode=False)
    jobs: tp.List[execution.MockedTimedJob] = []
    for k in range(10):
        jobs.append(executor.submit(func, k, 0))
    results: tp.List[float] = []
    while jobs:
        finished = [j for j in jobs if j.done()]
        np.testing.assert_equal(len(finished), 1)
        results.append(finished[0].result())
        jobs.remove(finished[0])
    np.testing.assert_array_equal(results, expected)


def test_mocked_steady_executor_time() -> None:
    func = Function()
    executor = execution.MockedTimedExecutor(batch_mode=False)
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
    np.testing.assert_equal(sum(j.done() for j in jobs), 2)
    np.testing.assert_array_equal([j.release_time for j in executor._steady_priority_queue], [4, 5, 5])
    new_finished: tp.Optional[tp.List[execution.MockedTimedJob]] = None
    order: tp.List[int] = []
    # pylint: disable=unsubscriptable-object
    while new_finished is None or new_finished:
        if new_finished is not None:
            assert len(new_finished) == 1, f'Weird list: {new_finished}'
            order.append(jobs.index(new_finished[0]))
            new_finished[0].result()
        new_finished = [j for j in jobs if j.done() and not j._is_read]
    np.testing.assert_array_equal(order, [2, 0, 3])


def test_batch_executor_time() -> None:
    func = Function()
    executor = execution.MockedTimedExecutor(batch_mode=True)
    jobs = [executor.submit(func, k, 0) for k in range(3)]
    np.testing.assert_equal([j.release_time for j in jobs], [5, 4, 3])
    np.testing.assert_equal([j.done() for j in jobs], [True, True, True])
    for job in jobs:
        job.result()
    np.testing.assert_equal(executor.time, 5)
    job = executor.submit(func, 0, 0)
    np.testing.assert_equal(job.release_time, 10)


def test_functionlib_delayed_job() -> None:
    np.random.seed(None)
    func = ArtificialFunction("DelayedSphere", 2)
    func([0, 0])  # trigger init
    executor = execution.MockedTimedExecutor(batch_mode=False)
    x0 = func.transform_var._transforms[0].translation  # optimal value
    job0 = executor.submit(func, x0)
    job1 = executor.submit(func, x0 + 1.)
    assert job0.release_time == 0
    assert job1.release_time > 0
