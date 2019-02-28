# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import time
import inspect
import warnings
from numbers import Real
from collections import deque
from typing import Optional, Tuple, Callable, Any, Dict, List, Union, NamedTuple, Deque, Type
import numpy as np
from ..common.typetools import ArrayLike, JobLike, ExecutorLike
from .. import instrumentation as instru
from ..common.tools import Sleeper
from ..common.decorators import Registry
from . import utils


registry = Registry()

_OptimCallBack = Union[Callable[["Optimizer", ArrayLike, float], None], Callable[["Optimizer"], None]]


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
        dimension = int(dimension)
        self.dimension = dimension
        self.name = self.__class__.__name__  # printed name in repr
        # keep a record of evaluations, and current bests which are updated at each new evaluation
        self.archive: Union[utils.Archive, Dict[Tuple[float, ...], utils.Value]] = {}
        self.current_bests = {x: utils.Point(tuple(0. for _ in range(dimension)), utils.Value(np.inf))
                              for x in ["optimistic", "pessimistic", "average"]}
        # instance state
        self._num_ask = 0
        self._num_tell = 0
        self._callbacks: Dict[str, List[Any]] = {}
        # to make optimize function stoppable halway through
        self._running_jobs: List[Tuple[ArrayLike, JobLike]] = []
        self._finished_jobs: Deque[Tuple[ArrayLike, JobLike]] = deque()

    @property
    def num_ask(self) -> int:
        return self._num_ask

    @property
    def num_tell(self) -> int:
        return self._num_tell

    @property
    def num_suggestions(self) -> int:
        warnings.warn("Use num_ask property instead", DeprecationWarning)
        return self.num_ask

    @property
    def num_evaluations(self) -> int:
        warnings.warn("Use num_tell property instead", DeprecationWarning)
        return self.num_tell

    def __repr__(self) -> str:
        return f"Instance of {self.name}(dimension={self.dimension}, budget={self.budget}, num_workers={self.num_workers})"

    def register_callback(self, name: str, callback: _OptimCallBack) -> None:
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
        # call callbacks for logging etc...
        for callback in self._callbacks.get("tell", []):
            callback(self, x, value)
        if not isinstance(value, Real):
            raise TypeError(f'"tell" method only supports float values but the passed value was: {value} (type: {type(value)}.')
        if np.isnan(value) or value == np.inf:
            warnings.warn(f"Updating fitness with {value} value")
        x = tuple(x)
        if x not in self.archive:
            self.archive[x] = utils.Value(value)  # better not to stock the position as a Point (memory)
        else:
            self.archive[x].add_evaluation(value)
        # update current best records
        # this may have to be improved if we want to keep more kinds of best values
        for name in ["optimistic", "pessimistic", "average"]:
            if x == self.current_bests[name].x:   # reboot
                if isinstance(self.archive, utils.Archive):
                    # currently, cast to tuple for compatibility reason (comparing tuples and np.ndarray fails)
                    y: ArrayLike = tuple(np.frombuffer(
                        min(self.archive.bytesdict, key=lambda x, n=name: self.archive.bytesdict[x].get_estimation(n))))
                else:
                    y = min(self.archive, key=lambda x, n=name: self.archive[x].get_estimation(n))
                # rebuild best point may change, and which value did not track the updated value anyway
                self.current_bests[name] = utils.Point(y, self.archive[y])
            else:
                if self.archive[x].get_estimation(name) <= self.current_bests[name].get_estimation(name):
                    self.current_bests[name] = utils.Point(x, self.archive[x])
                if not (np.isnan(value) or value == np.inf):
                    assert self.current_bests[name].x in self.archive, "Best value should exist in the archive"
        self._internal_tell(x, value)
        self._num_tell += 1

    def ask(self) -> Tuple[float, ...]:
        """Provides a point to explore.
        This function can be called multiple times to explore several points in parallel
        """
        # call callbacks for logging etc...
        for callback in self._callbacks.get("ask", []):
            callback(self)
        suggestion = self._internal_ask()
        assert suggestion is not None, f"{self.__class__.__name__}._internal_ask method returned None instead of a point."
        self._num_ask += 1
        return suggestion

    def provide_recommendation(self) -> Tuple[float, ...]:
        """Provides the best point to use as a minimum, given the budget that was used
        """
        return self.recommend()  # duplicate method

    def recommend(self) -> Tuple[float, ...]:
        """Provides the best point to use as a minimum, given the budget that was used
        """
        return self._internal_provide_recommendation()

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
        if executor is None:
            executor = utils.SequentialExecutor()  # defaults to run everything locally and sequentially
            if self.num_workers > 1:
                warnings.warn(f"num_workers = {self.num_workers} > 1 is suboptimal when run sequentially", InefficientSettingsWarning)
        tmp_runnings: List[Tuple[ArrayLike, JobLike]] = []
        tmp_finished: Deque[Tuple[ArrayLike, JobLike]] = deque()
        # go
        sleeper = Sleeper()  # manages waiting time depending on execution time of the jobs
        remaining_budget = self.budget - self.num_ask
        first_iteration = True
        while remaining_budget or self._running_jobs or self._finished_jobs:
            # # # # # Update optimizer with finished jobs # # # # #
            # this is the first thing to do when resuming an existing optimization run
            # process finished
            if self._finished_jobs:
                if (remaining_budget or sleeper._start is not None) and not first_iteration:
                    # ignore stop if no more suggestion is sent
                    # this is an ugly hack to avoid warnings at the end of steady mode
                    sleeper.stop_timer()
                while self._finished_jobs:
                    x, job = self._finished_jobs[0]
                    self.tell(x, job.result())
                    self._finished_jobs.popleft()  # remove it after the tell to make sure it was indeed "told" (in case of interruption)
                    if verbosity:
                        print(f"Updating fitness with value {job.result()}")
                if verbosity:
                    print(f"{remaining_budget} remaining budget and {len(self._running_jobs)} running jobs")
                    if verbosity > 1:
                        print("Current pessimistic best is: {}".format(self.current_bests["pessimistic"]))
            elif not first_iteration:
                sleeper.sleep()
            # # # # # Start new jobs # # # # #
            if not batch_mode or not self._running_jobs:
                new_sugg = min(remaining_budget, self.num_workers - len(self._running_jobs))
                if verbosity and new_sugg:
                    print(f"Launching {new_sugg} jobs with new suggestions")
                for _ in range(new_sugg):
                    x = self.ask()
                    self._running_jobs.append((x, executor.submit(objective_function, x)))
                if new_sugg:
                    sleeper.start_timer()
            remaining_budget = self.budget - self.num_ask
            # split (repopulate finished and runnings in only one loop to avoid
            # weird effects if job finishes in between two list comprehensions)
            tmp_runnings, tmp_finished = [], deque()
            for x_job in self._running_jobs:
                (tmp_finished if x_job[1].done() else tmp_runnings).append(x_job)
            self._running_jobs, self._finished_jobs = tmp_runnings, tmp_finished
            first_iteration = False
        return self.provide_recommendation()


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
        if (time.time() - self._last_time) > self._num_sec or (self._num_eval and not optimizer.num_tell % self._num_eval):
            x = optimizer.provide_recommendation()
            point = x if x not in optimizer.archive else utils.Point(x, optimizer.archive[x])
            print(f"After {optimizer.num_tell}, recommendation is {point}")


class OptimizerFamily:
    """Factory/family of optimizers.
    This class only provides a very general pattern for it and enable its instances for use in
    benchmarks.
    """
    # this class will probably evolve in the near future
    # the naming pattern is not yet very clear, better ideas are welcome

    # optimizer qualifiers
    recast = False  # algorithm which were not designed to work with the suggest/update pattern
    one_shot = False  # algorithm designed to suggest all budget points at once
    no_parallelization = False  # algorithm which is designed to run sequentially only
    hashed = False

    def __init__(self, **kwargs: Any) -> None:  # keyword only, to be as explicit as possible
        self._kwargs = kwargs
        params = ", ".join(f"{x}={y!r}" for x, y in sorted(kwargs.items()))
        self._repr = f"{self.__class__.__name__}({params})"  # ugly hack

    def __repr__(self) -> str:
        return self._repr

    def with_name(self, name: str, register: bool = False) -> 'OptimizerFamily':
        self._repr = name
        if register:
            registry.register_name(name, self)
        return self

    def __call__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> Optimizer:
        raise NotImplementedError


class ParametrizedFamily(OptimizerFamily):
    """This is a special case of an optimizer family for which the family instance serves to*
    hold the parameters.
    This class assumes that the attributes are set to the init parameters.
    See oneshot.py for examples
    """

    _optimizer_class: Optional[Type[Optimizer]] = None

    def __init__(self) -> None:
        defaults = {x: y.default for x, y in inspect.signature(self.__class__.__init__).parameters.items()
                    if x not in ["self", "__class__"]}
        diff = set(defaults.keys()).symmetric_difference(self.__dict__.keys())
        if diff:  # this is to help durring development
            raise RuntimeError(f"Mismatch between attributes and arguments of ParametrizedFamily: {diff}")
        # only print non defaults
        different = {x: self.__dict__[x] for x, y in defaults.items() if y != self.__dict__[x] and not x.startswith("_")}
        super().__init__(**different)

    def __call__(self, dimension: int, budget: Optional[int] = None, num_workers: int = 1) -> Optimizer:
        assert self._optimizer_class is not None
        run = self._optimizer_class(dimension=dimension, budget=budget, num_workers=num_workers)  # pylint: disable=not-callable
        assert hasattr(run, "_parameters")
        assert isinstance(run._parameters, self.__class__)  # type: ignore
        run._parameters = self  # type: ignore
        run.name = repr(self)
        return run


class ArgPoint(NamedTuple):
    """Handle for args and kwargs arguments, keeping
    the initial data in memory.
    """
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    data: ArrayLike


class IntrumentedOptimizer:
    """Optimizer which yields "ArgPoint"s instead of data points (np.ndarray).
    ArgPoint structure directly provides args and kwargs to input to the function you
    mean to optimize.
    """

    def __init__(self, optimizer: Optimizer, instrumentation: instru.Instrumentation) -> None:
        assert optimizer.dimension == instrumentation.dimension
        self._optimizer = optimizer
        self.instrumentation = instrumentation

    def ask(self) -> ArgPoint:
        x = self._optimizer.ask()
        args, kwargs = self.instrumentation.data_to_arguments(x)
        return ArgPoint(args, kwargs, x)

    def provide_recommendation(self) -> ArgPoint:
        x = self._optimizer.provide_recommendation()
        args, kwargs = self.instrumentation.data_to_arguments(x)
        return ArgPoint(args, kwargs, x)

    def tell(self, point: ArgPoint, value: float) -> None:
        assert isinstance(point, ArgPoint), '"tell" can only receive an ArgPoint'
        self._optimizer.tell(point.data, value)

    def optimize(self, objective_function: Callable[..., float],
                 executor: Optional[ExecutorLike] = None,
                 batch_mode: bool = False,
                 verbosity: int = 0) -> ArgPoint:
        # for now, instrument the function and optimize
        # this should be updated eventually to take benefit of the information
        # provided by the instumentation
        ifunc = instru.InstrumentedFunction(objective_function, *self.instrumentation.args,
                                            **self.instrumentation.kwargs)
        self._optimizer.optimize(ifunc, executor=executor, batch_mode=batch_mode, verbosity=verbosity)
        return self.provide_recommendation()
