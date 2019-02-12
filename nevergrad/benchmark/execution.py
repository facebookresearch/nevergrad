# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import heapq
from typing import List, Callable, Any, NamedTuple
from ..functions.base import PostponedObject  # this object only serves to provide delays that the executor must use to order jobs


class MockedSteadyJob:
    """Job returned by the MockedSteadyExecutor, with the usual
    "done()" and "result()" methods.
    """

    def __init__(self, value: float, executor: "MockedSteadyExecutor") -> None:
        self._value = value
        self._done = False
        self._is_read = False
        self._executor = executor

    def done(self) -> bool:
        return self._done

    def result(self) -> float:
        """Return the result if "done()" is true, and raises
        a RuntimeError otherwise.
        """
        if not self._done:
            raise RuntimeError("Asking result which is not ready")
        self._is_read = True
        self._executor.update_queue()
        return self._value

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}<value={self._value}, done={self._done}, read={self._is_read}>)'


class OrderedJobs(NamedTuple):
    """Handle for sorting jobs by release_time (or submission order in case of tie)
    """
    release_time: float
    order: int
    job: MockedSteadyJob


class MockedSteadyExecutor:
    """Executor that mocks a steady state, by only providing 1 job at a time which is done() while
    not having been "read" (i.e. "result()" method was not executed).
    This ensures we control the order of update of the optimizer for benchmarking.

    Additionally, "delays" can be provided by the function so that jobs are not "done()" by order of
    submission. To this end, callables must implement a "computation_time" method.
    """

    def __init__(self) -> None:
        self.priority_queue: List[OrderedJobs] = []
        self._order = 0
        self._time = 0.

    def submit(self, function: Callable, *args: Any, **kwargs: Any) -> MockedSteadyJob:
        if self.priority_queue:  # new job may come before the current "next" job
            self.priority_queue[0].job._done = False
        value = function(*args, **kwargs)
        job = MockedSteadyJob(value, self)
        # compute the delay and add to queue
        delay = 0.
        if isinstance(function, PostponedObject):
            delay = max(0, function.get_postponing_delay((args, kwargs), value))
        heapq.heappush(self.priority_queue, OrderedJobs(self._time + delay, self._order, job))
        # update order and "next" finished job
        self._order += 1
        self.priority_queue[0].job._done = True
        return job

    def update_queue(self) -> None:
        """Called whenever a result is read, so as to activate the next result in line
        """
        while self.priority_queue and self.priority_queue[0].job._is_read:
            self._time = heapq.heappop(self.priority_queue).release_time
            if self.priority_queue:
                self.priority_queue[0].job._done = True
