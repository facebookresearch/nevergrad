# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Nevergrad's API provides ask/tell, letting the user run the objective function
in their own way. Some libraries for optimization take the objective function
as a parameter and run it themselves.
This module is a system for adapting such a library to be used in nevergrad,
by running the optimization on a background thread.
The file recastlib.py has some uses of it.
For instructions for doing this for another library, see
SequentialRecastOptimizer.
"""


import warnings
import threading
import queue
import numpy as np
import nevergrad.common.typing as tp
from nevergrad.common.errors import NevergradError
from nevergrad.parametrization import parameter as p
from . import base
from .base import IntOrParameter


class StopOptimizerThread(Exception):
    pass


class TooManyAskError(NevergradError):
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

    def run(self) -> None:
        """Starts the thread and run the "caller" function argument on
        the fake callable, which posts messages and awaits for their answers.
        """
        try:
            self.output = self._caller(self._fake_callable, *self._args, **self._kwargs)
        except StopOptimizerThread:  # gracefully stopping the thread
            self.messages_ask.put(ValueError("Optimization told to finish"))
        except Exception as e:  # pylint: disable=broad-except
            self.messages_ask.put(e)
            self.error = e
        else:
            self.messages_ask.put(None)

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

    An optimization is performed by a third-party library in a background thread. This communicates
    with the main thread using two queue objects. Specifically:

        messages_ask is filled by the background thread with a candidate (or batch of candidates)
        it wants evaluated for, or None if the background thread is over, or an Exception
        which needs to be raised to the user.

        messages_tell supplies the background thread with a value to return from the fake function.
        A value of None means the background thread is no longer relevant and should exit.
    """

    recast = True

    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int] = None,
        num_workers: int = 1,
    ) -> None:
        super().__init__(parametrization, budget, num_workers=num_workers)
        self._messaging_thread: tp.Optional[MessagingThread] = None  # instantiate at runtime

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

    def _check_error(self) -> None:
        if self._messaging_thread is not None:
            if self._messaging_thread.error is not None:
                raise RuntimeError(
                    f"Recast optimizer raised an error:\n{self._messaging_thread.error}"
                ) from self._messaging_thread.error

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
    """Recast Optimizer which cannot deal with parallelization

    There can only be one worker. Each ask must be followed by
    a tell.

    A simple usage is that you have a library which can minimize
    a function which returns a scalar.
    Just make an optimizer inheriting from this class, and inplement
    get_optimization_function to return a callable which runs the
    optimization, taking the objective as its only parameter. The
    callable must not have any references to the optimizer itself.
    (This avoids a reference cycle between the background thread and
    the optimizer, aiding cleanup.) It can have a weakref though.

    If you want your optimizer instance to be picklable, we have to
    store every candidate during optimization, which may use a lot
    of memory. This lets us replay the optimization when
    unpickling. We only do this if you ask for it. To enable:
        - The optimization must be reproducible, asking for the same
          candidates every time. If you need a seed from nevergrad's
          generator, you can't necessarily generate this again after
          unpickling. One solution is to store it in self, if it is
          not there yet, in the body of get_optimization_function.

          As in general in nevergrad, do not set the seed from the
          RNG in your own __init__ because it will cause surprises
          to anyone re-seeding your parametrization after init.
        - The user must call enable_pickling() after initializing
          the optimizer instance.
    """

    # pylint: disable=abstract-method

    no_parallelization = True

    def __init__(
        self,
        parametrization: IntOrParameter,
        budget: tp.Optional[int],
        num_workers: int = 1,
    ) -> None:
        super().__init__(parametrization=parametrization, budget=budget, num_workers=num_workers)
        self._enable_pickling = False
        self.replay_archive_tell: tp.List[p.Parameter] = []

    def enable_pickling(self):
        """Make the optimizer store its history of tells, so
        that it can be serialized.
        """
        if self.num_ask != 0:
            raise ValueError("Can only enable pickling before all asks.")
        self._enable_pickling = True

    def _internal_ask_candidate(self) -> p.Parameter:
        """Reads messages from the thread in which the underlying optimization function is running
        New messages are sent as "ask".
        """
        if self._messaging_thread is None:
            self._messaging_thread = MessagingThread(self.get_optimization_function())
        alive = self._messaging_thread.is_alive()
        if alive:
            point = self._messaging_thread.messages_ask.get()
            if isinstance(point, Exception):
                raise point
        if not alive or point is None:  # In case the algorithm stops before the budget is elapsed.
            warnings.warn(
                "Underlying optimizer has already converged, returning random points",
                base.errors.FinishedUnderlyingOptimizerWarning,
            )
            self._check_error()
            data = self._rng.normal(0, 1, self.dimension)
            return self.parametrization.spawn_child().set_standardized_data(data)
        candidate = self.parametrization.spawn_child().set_standardized_data(point)
        return candidate

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: float) -> None:
        """Returns value for a point which was "asked"
        (none asked point cannot be "tell")
        """
        assert self._messaging_thread is not None, 'Start by using "ask" method, instead of "tell" method'
        if not self._messaging_thread.is_alive():  # optimizer is done
            self._check_error()
            return
        if self._enable_pickling:
            self.replay_archive_tell.append(candidate)
        self._messaging_thread.messages_tell.put(self._post_loss(candidate, loss))

    def __getstate__(self):
        if not self._enable_pickling:
            raise ValueError("If you want picklability you should have asked for it")
        thread = self._messaging_thread
        self._messaging_thread = None
        state = self.__dict__.copy()
        self._messaging_thread = thread
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not self._enable_pickling:
            raise ValueError("Cannot unpickle the unpicklable")

        # We temporarily unset _enable_pickling so that the replays do not
        # get archived again.
        self._enable_pickling = False
        for i, candidate in enumerate(self.replay_archive_tell):
            new_candidate = self._internal_ask_candidate()
            norm = np.linalg.norm(new_candidate.get_standardized_data(reference=candidate))
            # Check that the replay wants the same value as we had the first time.
            # If an error is raised here then you might want to
            # check the reproducibility of your optimizer.
            if norm > 0.00001:
                raise RuntimeError(f"Mismatch in replay at index {i} of {len(self.replay_archive_tell)}.")
            self._internal_tell_candidate(candidate, candidate.loss)

        if self.num_ask > self.num_tell:
            self._internal_ask_candidate()

        self._enable_pickling = True


class BatchRecastOptimizer(RecastOptimizer):
    """Recast optimizer where points to evaluate are provided in batches
    and stored by the optimizer to be asked and told on. The fake_callable
    is only brought into action every 'batch size' number of asks and tells
    instead of every ask and tell. This opens up the optimizer to
    parallelism.

    Note
    ----
    You have to complete a batch before you start a new one so parallelism
    is only possible within batches i.e. if a batch size is 100 and you have
    done 100 asks, you must do 100 tells before you ask again but you could do
    those 100 asks and tells in parallel. To find out if you can perform an ask
    at any given time call self.can_ask.
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
        if self._messaging_thread is None:
            self._messaging_thread = MessagingThread(self.get_optimization_function())
        if self._current_batch:
            return self._current_batch.pop()
        # if there are any points in the previous batch that haven't been told on, you cannot update the current batch.
        if not self.can_ask():
            raise TooManyAskError(
                "You can't get a new batch until the old one has been fully told on. See docstring for more info."
            )
        alive = self._messaging_thread.is_alive()
        # wait for a message
        if alive:
            points = self._messaging_thread.messages_ask.get()
            if isinstance(points, Exception):
                raise points
        if not alive or points is None:  # In case the algorithm stops before the budget is elapsed.
            warnings.warn(
                "Underlying optimizer has already converged, returning random points",
                base.errors.FinishedUnderlyingOptimizerWarning,
            )
            self._check_error()
            data = self._rng.normal(0, 1, self.dimension)
            return self.parametrization.spawn_child().set_standardized_data(data)

        # We have a new batch
        self.batch_size = len(points)
        self._current_batch = [
            self.parametrization.spawn_child().set_standardized_data(point) for point in points
        ]
        self._batch_losses = [None] * len(points)  # type: ignore
        # map each point to an index in preparation to build loss array in tell
        self.indices = {candidate.uid: i for i, candidate in enumerate(self._current_batch)}
        return self._current_batch.pop()

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
            self._messaging_thread.messages_tell.put(np.array(self._batch_losses))
            self._batch_losses = []
            self._tell_counter = 0

    def minimize(
        self,
        objective_function: tp.Callable[..., tp.Loss],
        executor: tp.Optional[tp.ExecutorLike] = None,
        batch_mode: bool = False,
        verbosity: int = 0,
        constraint_violation: tp.Any = None,
    ) -> p.Parameter:
        raise NotImplementedError("This optimizer isn't supported by the way minimize works by default.")

    def can_ask(self) -> bool:
        """Returns whether the optimizer is able to perform another ask,
        either because there are points left in the current batch to ask
        or you are ready for a new batch (You have asked and told on every
        point in the last batch.)
        """
        return len(self.indices) == 0 or len(self._current_batch) > 0
