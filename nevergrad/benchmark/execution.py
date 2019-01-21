import heapq
from typing import List, Callable, Any, NamedTuple


class MockedSteadyJob:

    def __init__(self, value: float, executor: "MockedSteadyExecutor") -> None:
        self._value = value
        self._done = False
        self._is_read = False
        self._executor = executor

    def done(self) -> bool:
        return self._done

    def result(self) -> float:
        if not self._done:
            raise RuntimeError("Asking result which is not ready")
        self._is_read = True
        self._executor.update_queue()
        return self._value

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}<value={self._value}, done={self._done}, read={self._is_read}>)'


class OrderedJobs(NamedTuple):
    release_time: float
    order: int
    job: MockedSteadyJob


class MockedSteadyExecutor:

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
        delay = 0
        if hasattr(function, "computation_time"):
            delay = max(0, function.computation_time((args, kwargs), value))  # type: ignore
        heapq.heappush(self.priority_queue, OrderedJobs(self._time + delay, self._order, job))
        # update order and "next" finished job
        self._order += 1
        self.priority_queue[0].job._done = True
        return job

    def update_queue(self) -> None:
        while self.priority_queue and self.priority_queue[0].job._is_read:
            self._time = heapq.heappop(self.priority_queue).release_time
            if self.priority_queue:
                self.priority_queue[0].job._done = True
