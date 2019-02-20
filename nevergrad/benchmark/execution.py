# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import heapq
from collections import deque
from typing import List, Callable, Any, NamedTuple, Tuple, Dict, Optional, Deque
from ..functions.base import PostponedObject  # this object only serves to provide delays that the executor must use to order jobs


class MockedTimedJob:
    """Job returned by the MockedTimedExecutor, with the usual
    "done()" and "result()" methods.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, func: Callable[..., Any], args: Tuple[Any, ...], kwargs: Dict[str, Any],
                 executor: "MockedTimedExecutor") -> None:
        self._executor = executor
        self._time = executor.time  # time at instantiation
        # function
        self._func = func
        self._args = args
        self._kwargs = kwargs
        # job specific
        self._output: Any = None
        self._delay: Optional[float] = None  # float when properly job is computed
        self._done = False
        self._is_read = False  # for the record

    @property
    def release_time(self) -> float:
        return self._get_delay() + self._time

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
        self._executor.notify_read(self)
        return self._output

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}<value={self._output}, done={self._done}, read={self._is_read}>)'


class OrderedJobs(NamedTuple):
    """Handle for sorting jobs by release_time (or submission order in case of tie)
    """
    release_time: float
    order: int
    job: MockedTimedJob


class MockedTimedExecutor:
    """Executor that mocks a steady state, by only providing 1 job at a time which is done() while
    not having been "read" (i.e. "result()" method was not executed).
    This ensures we control the order of update of the optimizer for benchmarking.

    Additionally, "delays" can be provided by the function so that jobs are not "done()" by order of
    submission. To this end, callables must implement a "computation_time" method.
    """

    def __init__(self, batch_mode: bool = False) -> None:
        self._batch_mode = batch_mode
        self._to_be_processed: Deque[MockedTimedJob] = deque()
        self._steady_priority_queue: List[OrderedJobs] = []
        self._order = -1
        self._time = 0.

    @property
    def time(self) -> float:
        return self._time

    def submit(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> MockedTimedJob:
        job = MockedTimedJob(function, args, kwargs, self)
        self._to_be_processed.append(job)  # save for later processing
        return job

    def _process_submissions(self) -> None:
        if not self._to_be_processed:
            return  # don't bother
        if self._steady_priority_queue:
            self._steady_priority_queue[0].job._done = False
        # first pass: compute everything (this may take a long time, safer this way in case of interruption)
        for job in self._to_be_processed:
            job._get_delay()
        # second path: update
        while self._to_be_processed:
            job = self._to_be_processed[0]
            if self._batch_mode:
                self._time = max(self._time, job.release_time)
                job._done = True
            else:
                self._order += 1
                heapq.heappush(self._steady_priority_queue, OrderedJobs(job.release_time, self._order, job))
            self._to_be_processed.popleft()  # remove right after it is added to the heap queue
        if self._steady_priority_queue:
            self._steady_priority_queue[0].job._done = True

    def notify_read(self, job: MockedTimedJob) -> None:
        """Called whenever a result is read, so as to activate the next result in line
        """
        self._process_submissions()  # make sure everything is up to date
        if not self._batch_mode:
            expected = self._steady_priority_queue[0]
            assert job is expected.job, "Only first job should be read"
            self._time = expected.release_time
            heapq.heappop(self._steady_priority_queue)
            if self._steady_priority_queue:
                self._steady_priority_queue[0].job._done = True
