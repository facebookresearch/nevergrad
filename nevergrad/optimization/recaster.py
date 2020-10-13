# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import warnings
import threading
import numpy as np
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from . import base
from .base import IntOrParameter


class Message:
    """Straightforward class for passing parameters of a function and
    results to and from a thread.

    Parameters
    ----------
    *args: Any
    **kwargs: Any

    Note
    ----
    - "result" attribute is a property which records when it is "posted"
    (ie message has been received and processed and "result" is the answer, which is ready)
    - "meta" attribute is only there for more implementation specific usages.
    """

    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.meta: tp.Dict[str, tp.Any] = {}  # for none Thread caller purposes
        self._result: tp.Optional[tp.Any] = None
        self.done = False

    @property
    def result(self) -> tp.Any:
        if not self.done:
            raise RuntimeError("Result was not provided (not done)")
        return self._result

    @result.setter
    def result(self, value: tp.Any) -> None:
        self.done = True
        self._result = value

    def __repr__(self) -> str:
        return (f"<Message: args={self.args}, kwargs={self.kwargs}" +
                f" (result: {self.result})>" if self.done else ">")


class StopOptimizerThread(Exception):
    pass


class _MessagingThread(threading.Thread):
    """Thread that runs a function taking another function as input. Each call of the inner function
    creates a Message with fields args and kwargs and waits for the main thread to set the result
    attribute of the message

    Note
    ----
    This thread must be overlaid into another MessagingThread  because:
    - the threading part should hold no reference from outside (otherwise the destructors may wait each other)
    - the destructor cannot be implemented, hence there is no way to stop the thread automatically
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, caller: tp.Callable[..., tp.Any], *args: tp.Any, **kwargs: tp.Any) -> None:
        super().__init__()
        self.messages: tp.List[Message] = []
        self.call_count = 0
        self.error: tp.Optional[Exception] = None
        self._kill_order = False
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
            self.messages.clear()
        except Exception as e:  # pylint: disable=broad-except
            self.error = e

    def _fake_callable(self, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        """Appends a message in the messages attribute of the thread when
        the caller needs an evaluation, and wait for it to be provided
        to return it to the caller
        """
        self.call_count += 1
        mess = Message(*args, **kwargs)
        self.messages.append(mess)  # sends a message
        t0 = time.time()
        while not (mess.done or self._kill_order):  # waits for its answer
            time.sleep(self._last_evaluation_duration / 10.)
        self._last_evaluation_duration = np.clip(time.time() - t0, .0001, 1.)
        # sys.stdout.write(f"Received answer {repr(mess)}\n")
        # sys.stdout.flush()
        if self._kill_order:
            raise StopOptimizerThread("Received kill order")  # kill the thread gracefully if asked to do so
        self.messages.remove(mess)  # remove the message, which is not useful anymore
        return mess.result

    def stop(self) -> None:
        """Notifies the thread that it must stop
        """
        self._kill_order = True


class MessagingThread:
    """Encapsulate the inner thread, so that kill order is automatically called at deletion.
    """

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
    def messages(self) -> tp.List[Message]:
        return self._thread.messages

    def stop(self) -> None:
        self._thread.stop()

    def __del__(self) -> None:
        self.stop()  # del method of the thread class does not work


class FinishedUnderlyingOptimizerWarning(Warning):
    pass


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

    def __init__(self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1) -> None:
        super().__init__(parametrization, budget, num_workers=num_workers)
        self._messaging_thread: tp.Optional[MessagingThread] = None  # instantiate at runtime
        self._last_optimizer_duration = 0.0001

    def get_optimization_function(self) -> tp.Callable[[tp.Callable[..., tp.Any]], tp.ArrayLike]:
        """Return an optimization procedure function (taking a function to optimize as input)

        Note
        ----
        This optimization procedure must be a function or an object which is completely
        independent from self, otherwise deletion of the optimizer may hang indefinitely.
        """
        raise NotImplementedError("You should define your optimizer! Also, be very careful to avoid "
                                  " reference to this instance in the returned object")

    def _internal_ask_candidate(self) -> p.Parameter:
        """Reads messages from the thread in which the underlying optimization function is running
        New messages are sent as "ask".
        """
        if self._messaging_thread is None:
            self._messaging_thread = MessagingThread(self.get_optimization_function())
        # wait for a message
        messages: tp.List[Message] = []
        t0 = time.time()
        while not messages and self._messaging_thread.is_alive():
            messages = [m for m in self._messaging_thread.messages if not m.meta.get("asked", False)]
            if not messages:  # avoid waiting if messages at the first iteration
                time.sleep(self._last_optimizer_duration / 10.)
        self._last_optimizer_duration = np.clip(time.time() - t0, .0001, 1.)
        # case when the thread is dead (send random points)
        if not self._messaging_thread.is_alive():  # In case the algorithm stops before the budget is elapsed.
            warnings.warn("Underlying optimizer has already converged, returning random points",
                          FinishedUnderlyingOptimizerWarning)
            self._check_error()
            data = np.random.normal(0, 1, self.dimension)
            return self.parametrization.spawn_child().set_standardized_data(data)
        message = messages[0]  # take oldest message
        message.meta["asked"] = True  # notify that it has been asked so that it is not selected again
        candidate = self.parametrization.spawn_child().set_standardized_data(message.args[0])
        message.meta["uid"] = candidate.uid
        return candidate

    def _check_error(self) -> None:
        if self._messaging_thread is not None:
            if self._messaging_thread.error is not None:
                raise RuntimeError(f"Recast optimizer raised an error:\n{self._messaging_thread.error}") from self._messaging_thread.error

    def _internal_tell_candidate(self, candidate: p.Parameter, value: float) -> None:
        """Returns value for a point which was "asked"
        (none asked point cannot be "tell")
        """
        x = candidate.get_standardized_data(reference=self.parametrization)
        assert self._messaging_thread is not None, 'Start by using "ask" method, instead of "tell" method'
        if not self._messaging_thread.is_alive():  # optimizer is done
            self._check_error()
            return
        messages = [m for m in self._messaging_thread.messages if m.meta.get("asked", False) and not m.done]
        messages = [m for m in messages if m.meta["uid"] == candidate.uid]
        if not messages:
            raise RuntimeError(f"No message for evaluated point {x}: {self._messaging_thread.messages}")
        messages[0].result = value  # post the value, and the thread will deal with it

    def _internal_tell_not_asked(self, candidate: p.Parameter, value: float) -> None:
        raise base.TellNotAskedNotSupportedError

    def _internal_provide_recommendation(self) -> tp.Optional[tp.ArrayLike]:
        """Returns the underlying optimizer output if provided (ie if the optimizer did finish)
        else the best pessimistic point.
        """
        if (self._messaging_thread is not None and
                self._messaging_thread.output is not None):
            return self._messaging_thread.output  # type: ignore
        else:
            return None  # use default

    def __del__(self) -> None:
        # explicitly ask the thread to stop (better be safe :))
        if self._messaging_thread is not None:
            self._messaging_thread.stop()


class SequentialRecastOptimizer(RecastOptimizer):
    """Recast Optimizer which cannot deal with parallelization
    """
    # pylint: disable=abstract-method

    no_parallelization = True
