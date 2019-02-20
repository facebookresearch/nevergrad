# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import heapq
from typing import List, Callable, Any, NamedTuple, Tuple, Dict, Optional
from ..functions.base import PostponedObject  # this object only serves to provide delays that the executor must use to order jobs


class MockedTimedJob:
    """Job returned by the MockedTimedExecutor, with the usual
    "done()" and "result()" methods.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, func: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any],
                 executor: "MockedSteadyExecutor") -> None:
        # function
        self._func = func
        self._args = args
        self._kwargs = kwargs
        # job specific
        self._output: Any = None
        self._delay: Optional[float] = None  # float when properly job is computed
        self._done = False
        self._is_read = False
        self._executor = executor

    def done(self) -> bool:
        self._executor._process_submissions()
        return self._done

    def _get_delay(self) -> float:
        if self._delay is None:
            self._output = self._func(*self._args, **self._kwargs)
            # compute the delay and add to queue
            self._delay = 1.
            if isinstance(self._func, PostponedObject):
                self._delay = max(0, self._func.get_postponing_delay((self._args, self._kwargs), self._output))
        return self._delay

    def result(self) -> Any:
        """Return the result if "done()" is true, and raises
        a RuntimeError otherwise.
        """
        if not self._done:
            raise RuntimeError("Asking result which is not ready")
        self._is_read = True
        self._executor.update_queue()
        return self._output

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}<value={self._output}, done={self._done}, read={self._is_read}>)'


class OrderedJobs(NamedTuple):
    """Handle for sorting jobs by release_time (or submission order in case of tie)
    """
    release_time: float
    order: int
    job: MockedTimedJob


class MockedSteadyExecutor:
    """Executor that mocks a steady state, by only providing 1 job at a time which is done() while
    not having been "read" (i.e. "result()" method was not executed).
    This ensures we control the order of update of the optimizer for benchmarking.

    Additionally, "delays" can be provided by the function so that jobs are not "done()" by order of
    submission. To this end, callables must implement a "computation_time" method.
    """

    def __init__(self) -> None:
        self._to_be_processed: List[MockedTimedJob] = []
        self._steady_priority_queue: List[OrderedJobs] = []
        self._order = 0
        self._time = 0.

    def submit(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> MockedTimedJob:
        if self._steady_priority_queue:  # new job may come before the current "next" job
            self._steady_priority_queue[0].job._done = False
        job = MockedTimedJob(function, args, kwargs, self)
        self._to_be_processed.append(job)
        return job

    def _process_submissions(self) -> None:
        if self._steady_priority_queue:
            self._steady_priority_queue[0].job._done = False
        # first pass: compute everything (this may take a long time, safer this way in case of interruption)
        for job in self._to_be_processed:
            job._get_delay()
        # second path: update
        for job in self._to_be_processed:
            delay = job._get_delay()
            heapq.heappush(self._steady_priority_queue, OrderedJobs(self._time + delay, self._order, job))
            # update order and "next" finished job
            self._order += 1
        if self._steady_priority_queue:
            self._steady_priority_queue[0].job._done = True
        self._to_be_processed.clear()

    def update_queue(self) -> None:
        """Called whenever a result is read, so as to activate the next result in line
        """
        self._process_submissions()
        while self._steady_priority_queue and self._steady_priority_queue[0].job._is_read:
            self._time = heapq.heappop(self._steady_priority_queue).release_time
            if self._steady_priority_queue:
                self._steady_priority_queue[0].job._done = True
