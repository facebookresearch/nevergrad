# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import time
import warnings
from typing import Optional, Tuple, Callable, Any, Dict, List, Union
import numpy as np
from ..common.typetools import ArrayLike, JobLike, ExecutorLike
from ..common.tools import Sleeper
from ..common.decorators import Registry
from . import utils


registry = Registry()


class InefficientSettingsWarning(RuntimeWarning):
    pass


class Optimizer(abc.ABC):  # pylint: disable=too-many-instance-attributes
    """Algorithm framework with 3 main functions:
    - suggest_exploration which provides points on which to evaluate the function to optimize
    - update_with_fitness_value which lets you provide the values associated to points
    - provide_recommendation which provides the best final value
    Typically, one would call suggest_exploration num_workers times, evaluate the
    function on these num_workers points in parallel, update with the fitness value when the
    evaluations is finished, and iterate until the budget is over. At the very end,
    one would call provide_recommendation for the estimated optimum.

    This class is abstract, it provides _internal equivalents for the 3 main functions,
    among which at least _internal_suggest_exploration must be overridden.

    Each optimizer instance should be used only once, with the initial provided budget

    Parameters
    ----------
    dimension: int
        dimension of the optimization space
    budget: int/None
        number of allowed evaluations
    num_workers: int
        number of evaluations which will be run in parallel at once
    """
    # pylint: disable=too-many-locals

    # optimizer qualifiers
    recast = False  # algorithm which were not designed to work with the suggest/update pattern
    one_shot = False  # algorithm designed to suggest all budget points at once
    no_parallelization = False  # algorithm which is designed to run sequentially only
    hashed = False

    def __init__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> None:
        if self.no_parallelization and num_workers > 1:
            raise ValueError(f"{self.__class__.__name__} does not support parallelization")
        self.num_workers = int(num_workers)
        self.budget = budget
        np.testing.assert_equal(dimension, int(dimension), f"Dimension must be an int")
        self.dimension = int(dimension)
        dimension = int(dimension)
        # keep a record of evaluations, and current bests which are updated at each new evaluation
        self.archive: Dict[Tuple[float, ...], utils.Value] = {}
        self.current_bests = {x: utils.Point(tuple(0. for _ in range(dimension)), utils.Value(np.inf))
                              for x in ["optimistic", "pessimistic"]}
        # instance state
        self._num_suggestions = 0
        self._num_evaluations = 0
        self._callbacks: Dict[str, List[Callable]] = {}

    @property
    def num_suggestions(self) -> int:
        return self._num_suggestions

    @property
    def num_evaluations(self) -> int:
        return self._num_evaluations

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"Instance of {self.name}(dimension={self.dimension}, budget={self.budget}, num_workers={self.num_workers})"

    def register_callback(self, name: str, callback: Union[Callable[["Optimizer", ArrayLike, float], None],
                                                           Callable[["Optimizer"], None]]) -> None:
        """Add a callback method called either when "tell" or "ask" are called, with the same
        arguments (including the optimizer / self). This can be useful for custom logging.

        Parameters
        ----------
        name: str
            name of the method to register the callback for (either "ask" or "tell")
        callback: callable
            a callable taking the same parameters as the method it is registered upon (including self)
        """
        assert name in ["ask", "tell"], f'Only "ask" and "tell" methods can have callbacks (not {name})'
        self._callbacks.setdefault(name, []).append(callback)

    def remove_all_callbacks(self) -> None:
        """Removes all registered callables
        """
        self._callbacks = {}

    def tell(self, x: ArrayLike, value: float) -> None:
        """Provides the optimizer with the evaluation of a fitness value at a point

        Parameters
        ----------
        x: tuple/np.ndarray
            point where the function was evaluated
        value: float
            value of the function
        """
        if np.isnan(value) or value == np.inf:
            warnings.warn(f"Updating fitness with {value} value")
        x = tuple(x)
        if x not in self.archive:
            self.archive[x] = utils.Value(value)  # better not to stock the position as a Point (memory)
        else:
            self.archive[x].add_evaluation(value)
        # update current best records
        # this may have to be improved if we want to keep more kinds of best values
        for name in ["optimistic", "pessimistic"]:
            if x == self.current_bests[name].x:   # reboot
                y: Tuple[float, ...] = min(self.archive, key=lambda x, n=name: self.archive[x].get_estimation(n))  # type: ignore
                # rebuild best point may change, and which value did not track the updated value anyway
                self.current_bests[name] = utils.Point(y, self.archive[y])
            else:
                if self.archive[x].get_estimation(name) <= self.current_bests[name].get_estimation(name):
                    self.current_bests[name] = utils.Point(x, self.archive[x])
                if not (np.isnan(value) or value == np.inf):
                    assert self.current_bests[name].x in self.archive, "Best value should exist in the archive"
        self._internal_tell(x, value)
        self._num_evaluations += 1
        # call callbacks for logging etc...
        for callback in self._callbacks.get("tell", []):
            callback(self, x, value)

    def ask(self) -> Tuple[float, ...]:
        """Provides a point to explore.
        This function can be called multiple times to explore several points in parallel
        """
        suggestion = self._internal_ask()
        self._num_suggestions += 1
        # call callbacks for logging etc...
        for callback in self._callbacks.get("ask", []):
            callback(self)
        return suggestion

    def provide_recommendation(self) -> Tuple[float, ...]:
        """Provides the best point to use as a minimum, given the budget that was used
        """
        return self._internal_provide_recommendation()

    # For compatibility. To be removed. TODO(oteytaud)
    def suggest_point(self) -> Tuple[float, ...]:
        warnings.warn("suggest_point should be converted to suggest_exploration", DeprecationWarning)
        return self.suggest_exploration()

    # Internal methods which can be overloaded (or must be, in the case of _internal_ask)
    def _internal_tell(self, x: ArrayLike, value: float) -> None:
        pass

    @abc.abstractmethod
    def _internal_ask(self) -> Tuple[float, ...]:
        raise NotImplementedError("Optimizer undefined.")

    def _internal_provide_recommendation(self) -> Tuple[float, ...]:
        return self.current_bests["pessimistic"].x

    def optimize(self, objective_function: Callable[[Any], float],
                 executor: Optional[ExecutorLike] = None,
                 batch_mode: bool = False,
                 verbosity: int = 0) -> Tuple[float, ...]:
        """Optimization (minimization) procedure

        Parameters
        ----------
        objective_function: callable
            A callable to optimize (minimize)
        executor: Executor
            An executor object, with method submit(callable, *args, **kwargs) and returning a Future-like object
            with methods done() -> bool and result() -> float. The executor role is to dispatch the execution of
            the jobs locally/on a cluster/with multithreading depending on the implementation.
            Eg: concurrent.futures.ThreadPoolExecutor
        batch_mode: bool
            when num_workers = n > 1, whether jobs are executed by batch (n function evaluations are launched,
            we wait for all results and relaunch n evals) or not (whenever an evaluation is finished, we launch
            another one)
        verbosity: int
            print information about the optimization (0: None, 1: fitness values, 2: fitness values and recommendation)
        callback: callable
            callable called on the optimizer (self) at the end of each iteration (for user specific logging, etc)

        Note
        ----
        for evaluation purpose and with the current implementation, it is better to use batch_mode=True
        """
        # pylint: disable=too-many-branches
        if self.budget is None:
            raise ValueError("Budget must be specified")
        num_workers = self.num_workers  # num_jobs ?
        budget = self.budget
        if executor is None:
            executor = utils.SequentialExecutor()  # defaults to run everything locally and sequentially
            if num_workers > 1:
                warnings.warn(f"num_workers = {num_workers} > 1 is suboptimal when run sequentially", InefficientSettingsWarning)
        # go
        runnings: List[Tuple[Any, JobLike]] = []
        finished: List[Tuple[Any, JobLike]] = []
        sleeper = Sleeper()  # manages waiting time depending on execution time of the jobs
        while budget or runnings:
            if not batch_mode or not runnings:
                new_sugg = min(budget, num_workers - len(runnings))
                if verbosity and new_sugg:
                    print(f"Launching {new_sugg} jobs with new suggestions")
                for _ in range(new_sugg):
                    x = self.ask()
                    runnings.append((x, executor.submit(objective_function, x)))
                    budget -= 1
                if new_sugg:
                    sleeper.start_timer()
            # split (repopulate finished and runnings in only one loop to avoid
            # weird effects if job finishes in between two list comprehensions)
            tmp = runnings
            runnings, finished = [], []
            for x_job in tmp:
                (finished if x_job[1].done() else runnings).append(x_job)
            # process finished
            if finished:
                sleeper.stop_timer()
                for x, job in finished:
                    self.tell(x, job.result())
                    if verbosity:
                        print(f"Updating fitness with value {job.result()}")
                if verbosity:
                    print(f"{budget} remaining budget and {len(runnings)} running jobs")
                    if verbosity > 1:
                        print("Current pessimistic best is: {}".format(self.current_bests["pessimistic"]))
            else:
                sleeper.sleep()
        return self.provide_recommendation()

    # the following functions are there for compatibility reasons only and should not be used

    def update_with_fitness_value(self, x: ArrayLike, value: float) -> None:
        warnings.warn("Use 'tell' instead of 'update_with_fitness_value' (will fail in December)", DeprecationWarning)  # TODO remove
        self.tell(x, value)

    def suggest_exploration(self) -> Tuple[float, ...]:
        warnings.warn("Use 'ask' instead of 'suggest_exploration' (will fail in December)", DeprecationWarning)  # TODO remove
        return self.ask()


class OptimizationPrinter:
    """Printer to register as callback in an optimizer, for printing
    best point regularly.

    Parameters
    ----------
    num_eval: int
        max number of evaluation before performing another print
    num_sec: float
        max number of seconds before performing another print
    """

    def __init__(self, num_eval: int = 0, num_sec: float = 60) -> None:
        self._num_eval = max(0, int(num_eval))
        self._last_time: Optional[float] = None
        self._num_sec = num_sec

    def __call__(self, optimizer: Optimizer, *args: Any, **kwargs: Any) -> None:
        if self._last_time is None:
            self._last_time = time.time()
        if (time.time() - self._last_time) > self._num_sec or (self._num_eval and not optimizer.num_evaluations % self._num_eval):
            x = optimizer.provide_recommendation()
            point = x if x not in optimizer.archive else utils.Point(x, optimizer.archive[x])
            print(f"After {optimizer.num_evaluations}, recommendation is {point}")
