# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import threading
import queue
import numpy as np
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from . import base
from .base import IntOrParameter


class StopOptimizerThread(Exception):
    pass


class _MessagingThread(threading.Thread):
    """Thread that runs a function taking another function as input. Each call of the inner function
    adds the point given by the algorithm into the ask queue and then blocks until the main thread sends
    the result back into the tell queue.

    Note
    ----
    This thread must be overlaid into another MessagingThread  because:
    - the threading part should hold no reference from outside (otherwise the destructors may wait each other)
    - the destructor cannot be implemented, hence there is no way to stop the thread automatically
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, caller: tp.Callable[..., tp.Any], *args: tp.Any, **kwargs: tp.Any) -> None:
        super().__init__()
        self.messages_ask: tp.Any = queue.Queue()
        self.messages_tell: tp.Any = queue.Queue()
        self.call_count = 0
        self.error: tp.Optional[Exception] = None
        self._caller = caller
        self._args = args
        self._kwargs = kwargs
        self.output: tp.Optional[tp.Any] = None  # TODO add a "done" attribute ?
        self._last_evaluation_duration = 0.0001

    def run(self) -> None:
        """Starts the thread and run the "caller" function argument on
        the fake callable, which posts messages and awaits for their answers.
        """
        try:
            self.output = self._caller(self._fake_callable, *self._args, **self._kwargs)
        except StopOptimizerThread:  # gracefully stopping the thread
            pass
        except Exception as e:  # pylint: disable=broad-except
            self.error = e

    def _fake_callable(self, *args: tp.Any) -> tp.Any:
        """
        Puts a new point into the ask queue to be evaluated on the
        main thread and blocks on get from tell queue until point
        is evaluated on main thread and placed into tell queue when
        it is then returned to the caller.
        """
        self.call_count += 1
        self.messages_ask.put(args[0])  # sends a message
        candidate = self.messages_tell.get()  # get evaluated message
        if candidate is None:
            raise StopOptimizerThread()
        return candidate

    def stop(self) -> None:
        """Notifies the thread that it must stop"""
        self.messages_tell.put(None)


class MessagingThread:
    """Encapsulate the inner thread, so that kill order is automatically called at deletion."""

    def __init__(self, caller: tp.Callable[..., tp.Any], *args: tp.Any, **kwargs: tp.Any) -> None:
        self._thread = _MessagingThread(caller, *args, **kwargs)
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    @property
    def output(self) -> tp.Any:
        return self._thread.output

    @property
    def error(self) -> tp.Optional[Exception]:
        return self._thread.error

    @property
    def messages_tell(self) -> tp.Any:
        return self._thread.messages_tell

    @property
    def messages_ask(self) -> tp.Any:
        return self._thread.messages_ask

    def stop(self) -> None:
        self._thread.stop()

    def __del__(self) -> None:
        self.stop()  # del method of the thread class does not work


class RecastOptimizer(base.Optimizer):
    """Base class for ask and tell optimizer derived from implementations with no ask and tell interface.
    The underlying optimizer implementation is a function which is supposed to call directly the function
    to optimize. It is tricked into optimizing a "fake" function in a thread:
    - calls to the fake functions are returned by the "ask()" interface
    - return values of the fake functions are provided to the thread when calling "tell(x, value)"

    Note
    ----
    These implementations are not necessarily robust. More specifically, one cannot "tell" any
    point which was not "asked" before.
    """

    recast = True

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget, num_workers=num_workers)
        self._messaging_thread: tp.Optional[MessagingThread] = None  # instantiate at runtime
        self._last_optimizer_duration = 0.0001

    def get_optimization_function(self) -> tp.Callable[[tp.Callable[..., tp.Any]], tp.Optional[tp.ArrayLike]]:
        """Return an optimization procedure function (taking a function to optimize as input)

        Note
        ----
        This optimization procedure must be a function or an object which is completely
        independent from self, otherwise deletion of the optimizer may hang indefinitely.
        """
        raise NotImplementedError(
            "You should define your optimizer! Also, be very careful to avoid "
            " reference to this instance in the returned object"
        )

    def _internal_ask_candidate(self) -> p.Parameter:
        """Reads messages from the thread in which the underlying optimization function is running
        New messages are sent as "ask".
        """
        if self._messaging_thread is None:
            self._messaging_thread = MessagingThread(self.get_optimization_function())
        # wait for a message
        if not self._messaging_thread.is_alive():  # In case the algorithm stops before the budget is elapsed.
            warnings.warn(
                "Underlying optimizer has already converged, returning random points",
                base.errors.FinishedUnderlyingOptimizerWarning,
            )
            self._check_error()
            data = self._rng.normal(0, 1, self.dimension)
            return self.parametrization.spawn_child().set_standardized_data(data)
        point = self._messaging_thread.messages_ask.get()
        candidate = self.parametrization.spawn_child().set_standardized_data(point)
        return candidate

    def _check_error(self) -> None:
        if self._messaging_thread is not None:
            if self._messaging_thread.error is not None:
                raise RuntimeError(
                    f"Recast optimizer raised an error:\n{self._messaging_thread.error}"
                ) from self._messaging_thread.error

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        """Returns value for a point which was "asked"
        (none asked point cannot be "tell")
        """
        assert self._messaging_thread is not None, 'Start by using "ask" method, instead of "tell" method'
        if not self._messaging_thread.is_alive():  # optimizer is done
            self._check_error()
            return
        self._messaging_thread.messages_tell.put(self._post_loss(candidate, loss))

    def _post_loss(self, candidate: p.Parameter, loss: float) -> tp.Loss:
        # pylint: disable=unused-argument
        """
        Posts the value, and the thread will deal with it.
        """
        return loss

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: float) -> None:
        raise base.errors.TellNotAskedNotSupportedError

    def _internal_provide_recommendation(self) -> tp.Optional[tp.ArrayLike]:
        """Returns the underlying optimizer output if provided (ie if the optimizer did finish)
        else the best pessimistic point.
        """
        if self._messaging_thread is not None and self._messaging_thread.output is not None:
            return self._messaging_thread.output
        else:
            return None  # use default

    def __del__(self) -> None:
        # explicitly ask the thread to stop (better be safe :))
        if self._messaging_thread is not None:
            self._messaging_thread.stop()


class SequentialRecastOptimizer(RecastOptimizer):
    """Recast Optimizer which cannot deal with parallelization"""

    # pylint: disable=abstract-method

    no_parallelization = True


class BatchRecastOptimizer(RecastOptimizer):
    """Recast optimizer where points to evaluate are provided in batches
    and stored by the optimizer to be asked and told on. The fake_callable
    is only brought into action every 'batch size' number of asks and tells
    instead of every ask and tell. This opens up the optimizer to
    parallelism.

    Note
    ----
    These implementations are not necessarily robust. You have to complete
    a batch before you start a new one so parallelism is only possible
    within batches i.e. if a batch size is 100 and you have done 100 asks,
    you must do 100 tells before you ask again but you could do those 100
    asks and tells in parallel.

    """

    # pylint: disable=abstract-method

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        super().__init__(parametrization, budget, num_workers=num_workers)
        self._current_batch: tp.List[p.Parameter] = []
        self._batch_losses: tp.List[tp.Loss] = []
        self._tell_counter = 0
        self.batch_size = 0
        self.indices: tp.Dict[str, int] = {}

    def _internal_ask_candidate(self) -> p.Parameter:
        """Reads messages from the thread in which the underlying optimization function is running
        New messages are sent as "ask".
        """
        # get a datapoint that is a random point in parameter space
        if self._messaging_thread is None:
            self._messaging_thread = MessagingThread(self.get_optimization_function())
        # wait for a message
        if not self._messaging_thread.is_alive():  # In case the algorithm stops before the budget is elapsed.
            warnings.warn(
                "Underlying optimizer has already converged, returning random points",
                base.errors.FinishedUnderlyingOptimizerWarning,
            )
            self._check_error()
            data = self._rng.normal(0, 1, self.dimension)
            return self.parametrization.spawn_child().set_standardized_data(data)
        # if no more points left in batch, wait for new batch from fake callable
        if not self._current_batch:
            if self.indices:
                raise RuntimeError(
                    "You can't ask on a new batch until the old one has been fully told on. See docstring for more info."
                )
            points = self._messaging_thread.messages_ask.get()
            self.batch_size = len(points)
            self._current_batch = [
                self.parametrization.spawn_child().set_standardized_data(point) for point in points
            ]
            self._batch_losses = [None] * len(points)  # type: ignore
            # map each point to an index in preparation to build loss array in tell
            self.indices = {candidate.uid: i for i, candidate in enumerate(self._current_batch)}
        candidate = self._current_batch.pop()
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        """Returns value for a point which was "asked"
        (none asked point cannot be "tell")
        """
        assert self._messaging_thread is not None, 'Start by using "ask" method, instead of "tell" method'
        if not self._messaging_thread.is_alive():  # optimizer is done
            self._check_error()
            return
        candidate_index = self.indices.pop(candidate.uid)
        self._batch_losses[candidate_index] = self._post_loss(candidate, loss)
        self._tell_counter += 1
        # if batch size number of tells since new batch, send array of losses to fake callable
        if self._tell_counter == self.batch_size:
            self._messaging_thread.messages_ask.put(np.array(self._batch_losses))
            self._batch_losses = []
            self._tell_counter = 0

    def minimize(
        self,
        objective_function: tp.Callable[..., tp.Loss],
        executor: tp.Optional[tp.ExecutorLike] = None,
        batch_mode: bool = False,
        verbosity: int = 0,
    ) -> p.Parameter:
        raise NotImplementedError("This optimizer isn't supported by the way minimize works by default.")
