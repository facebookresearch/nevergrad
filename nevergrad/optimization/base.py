# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import uuid
import time
import pickle
import inspect
import warnings
from pathlib import Path
from numbers import Real
from collections import deque
from typing import Optional, Tuple, Callable, Any, Dict, List, Union, Deque, Type, Set, TypeVar
import numpy as np
from ..common.typetools import ArrayLike, JobLike, ExecutorLike
from .. import instrumentation as instru
from ..common.tools import Sleeper
from ..common.decorators import Registry
from . import utils


registry = Registry[Union['OptimizerFamily', Type['Optimizer']]]()
_OptimCallBack = Union[Callable[["Optimizer", ArrayLike, float], None], Callable[["Optimizer"], None]]
X = TypeVar("X", bound="Optimizer")


def load(cls: Type[X], filepath: Union[str, Path]) -> X:
    """Loads a pickle file and checks that it contains an optimizer.
    The optimizer class is not always fully reliable though (e.g.: optimizer families) so the user is responsible for it.
    """
    filepath = Path(filepath)
    with filepath.open("rb") as f:
        opt = pickle.load(f)
    assert isinstance(opt, cls), f"You should only load {cls} with this method (found {type(opt)})"
    return opt


class InefficientSettingsWarning(RuntimeWarning):
    pass


class TellNotAskedNotSupportedError(NotImplementedError):
    """To be raised by optimizers which do not support the tell_not_asked interface.
    """


class Candidate:
    """Handle for args and kwargs arguments, keeping
    the initial data in memory.
    """

    def __init__(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], data: ArrayLike):
        self.args = args
        self.kwargs = kwargs
        self.data = np.array(data, copy=False)
        self.uuid = uuid.uuid4().hex
        self._meta: Dict[str, Any] = {}

    def __getitem__(self, ind: int) -> None:
        raise RuntimeError('Return type of "ask" is now a Candidate, use candidate.data[ind] '
                           '(rather than candidate[ind]) for the legacy behavior. '
                           'However, please update your code to use candidate.args and kwargs instead (see documentation).')

    def __array__(self) -> None:
        raise RuntimeError('Return type of "ask" is now a Candidate instead of an array. '
                           'You can use candidate.data to recover the data array as in the old versions. '
                           'However, please update your code to use args and kwargs instead (see documentation).')

    def __repr__(self) -> str:
        return f"Candidate(args={self.args}, kwargs={self.kwargs}, data={self.data})"

    def __str__(self) -> str:
        return f"Candidate(args={self.args}, kwargs={self.kwargs})"


class CandidateMaker:
    """Handle for creating Candidate instances easily

    Parameter
    ---------
    instrumentation: Instrumentation
        The instrumentation for converting from data space to arguments space.

    Note
    ----
    An instance of this class is linked to each optimizer (optimizer.create_candidate).
    Candidates can then easily be created through: optimizer.create_candidate.from_data(data)
    and/or optimizer.create_candidate.from_call(*args, **kwargs).
    """

    def __init__(self, instrumentation: instru.Instrumentation) -> None:
        self._instrumentation = instrumentation

    def __call__(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], data: ArrayLike) -> Candidate:
        return Candidate(args, kwargs, data)

    def from_call(self, *args: Any, **kwargs: Any) -> Candidate:
        """
        Parameters
        ----------
        *args, **kwargs: Any
            any arguments which match the instrumentation pattern.

        Returns
        -------
        Candidate:
            The corresponding candidate. Candidates have field "args" and "kwargs" which can be directly used
            on the function (objective_function(*candidate.args, **candidate.kwargs)).
        """
        data = self._instrumentation.arguments_to_data(*args, **kwargs)
        return Candidate(args, kwargs, data)

    def from_data(self, data: ArrayLike, deterministic: bool = False) -> Candidate:
        """Creates a Candidate, given a data from the optimization space

        Parameters
        ----------
        data: np.ndarray, List[float]...
            data from the optimization space
        deterministic: bool
            whether to sample arguments and kwargs from the distribution (when applicable) or
            create the most likely individual.

        Returns
        -------
        Candidate:
            The corresponding candidate. Candidates have field "args" and "kwargs" which can be directly used
            on the function (objective_function(*candidate.args, **candidate.kwargs)).
        """
        args, kwargs = self._instrumentation.data_to_arguments(data, deterministic=deterministic)
        return Candidate(args, kwargs, data)


class Optimizer:  # pylint: disable=too-many-instance-attributes
    """Algorithm framework with 3 main functions:
    - "ask()" which provides a candidate on which to evaluate the function to optimize
    - "tell(candidate, value)" which lets you provide the values associated to points
    - "provide_recommendation()" which provides the best final candidate
    Typically, one would call "ask()" num_workers times, evaluate the
    function on these num_workers points in parallel, update with the fitness value when the
    evaluations is finished, and iterate until the budget is over. At the very end,
    one would call provide_recommendation for the estimated optimum.

    This class is abstract, it provides _internal equivalents for the 3 main functions,
    among which at least _internal_ask has to be overridden.

    Each optimizer instance should be used only once, with the initial provided budget

    Parameters
    ----------
    instrumentation: int or Instrumentation
        either the dimension of the optimization space, or its instrumentation
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

    def __init__(self, instrumentation: Union[instru.Instrumentation, int],
                 budget: Optional[int] = None, num_workers: int = 1) -> None:
        if self.no_parallelization and num_workers > 1:
            raise ValueError(f"{self.__class__.__name__} does not support parallelization")
        self.num_workers = int(num_workers)
        self.budget = budget
        self.instrumentation = (instrumentation if isinstance(instrumentation, instru.Instrumentation) else
                                instru.Instrumentation(instru.var.Array(instrumentation)))
        if not self.dimension:
            raise ValueError("No variable to optimize in this instrumentation.")
        self.create_candidate = CandidateMaker(self.instrumentation)
        self.name = self.__class__.__name__  # printed name in repr
        # keep a record of evaluations, and current bests which are updated at each new evaluation
        self.archive: utils.Archive[utils.Value] = utils.Archive()  # dict like structure taking np.ndarray as keys and Value as values
        self.current_bests = {x: utils.Point(np.zeros(self.dimension, dtype=np.float), utils.Value(np.inf))
                              for x in ["optimistic", "pessimistic", "average"]}
        # pruning function, called at each "tell"
        # this can be desactivated or modified by each implementation
        self.pruning: Optional[Callable[[utils.Archive[utils.Value]], utils.Archive[utils.Value]]] = None
        self.pruning = utils.Pruning.sensible_default(num_workers=num_workers, dimension=self.instrumentation.dimension)
        # instance state
        self._asked: Set[str] = set()
        self._num_ask = 0
        self._num_tell = 0
        self._num_tell_not_asked = 0
        self._callbacks: Dict[str, List[Any]] = {}
        # to make optimize function stoppable halway through
        self._running_jobs: List[Tuple[Candidate, JobLike[float]]] = []
        self._finished_jobs: Deque[Tuple[Candidate, JobLike[float]]] = deque()

    @property
    def dimension(self) -> int:
        return self.instrumentation.dimension

    @property
    def num_ask(self) -> int:
        return self._num_ask

    @property
    def num_tell(self) -> int:
        return self._num_tell

    @property
    def num_tell_not_asked(self) -> int:
        return self._num_tell_not_asked

    def dump(self, filepath: Union[str, Path]) -> None:
        """Pickles the optimizer into a file.
        """
        filepath = Path(filepath)
        with filepath.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls: Type[X], filepath: Union[str, Path]) -> X:
        """Loads a pickle and checks that the class is correct.
        """
        return load(cls, filepath)

    def __repr__(self) -> str:
        inststr = f'{self.instrumentation:short}'
        return f"Instance of {self.name}(instrumentation={inststr}, budget={self.budget}, num_workers={self.num_workers})"

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

    def tell(self, candidate: Candidate, value: float) -> None:
        """Provides the optimizer with the evaluation of a fitness value for a candidate.

        Parameters
        ----------
        x: np.ndarray
            point where the function was evaluated
        value: float
            value of the function

        Note
        ----
        The candidate should generally be one provided by ask(), but can be also
        a non-asked candidate. To create a Candidate instance from args and kwargs,
        you can use optimizer.create_candidate.from_call(*args, **kwargs)
        """
        if not isinstance(candidate, Candidate):
            raise TypeError("'tell' must be provided with the candidate (use optimizer.create_candidate.from_call(*args, **kwargs)) "
                            "if you want to inoculate a point that as not been asked for")
        # call callbacks for logging etc...
        for callback in self._callbacks.get("tell", []):
            callback(self, candidate, value)
        self._update_archive_and_bests(candidate.data, value)
        if candidate.uuid in self._asked:
            self._internal_tell_candidate(candidate, value)
            self._asked.remove(candidate.uuid)
        else:
            self._internal_tell_not_asked(candidate, value)
            self._num_tell_not_asked += 1
        self._num_tell += 1

    def _update_archive_and_bests(self, x: ArrayLike, value: float) -> None:
        if not isinstance(value, (Real, float)):  # using "float" along "Real" because mypy does not understand "Real" for now Issue #3186
            raise TypeError(f'"tell" method only supports float values but the passed value was: {value} (type: {type(value)}.')
        if np.isnan(value) or value == np.inf:
            warnings.warn(f"Updating fitness with {value} value")
        if x not in self.archive:
            self.archive[x] = utils.Value(value)  # better not to stock the position as a Point (memory)
        else:
            self.archive[x].add_evaluation(value)
        # update current best records
        # this may have to be improved if we want to keep more kinds of best values
        for name in ["optimistic", "pessimistic", "average"]:
            if np.array_equal(x, self.current_bests[name].x):   # reboot
                y: bytes = min(self.archive.bytesdict, key=lambda z, n=name: self.archive.bytesdict[z].get_estimation(n))  # type: ignore
                # rebuild best point may change, and which value did not track the updated value anyway
                self.current_bests[name] = utils.Point(np.frombuffer(y), self.archive.bytesdict[y])
            else:
                if self.archive[x].get_estimation(name) <= self.current_bests[name].get_estimation(name):
                    self.current_bests[name] = utils.Point(x, self.archive[x])
                if not (np.isnan(value) or value == np.inf):
                    assert self.current_bests[name].x in self.archive, "Best value should exist in the archive"
        if self.pruning is not None:
            self.archive = self.pruning(self.archive)

    def ask(self) -> Candidate:
        """Provides a point to explore.
        This function can be called multiple times to explore several points in parallel

        Returns
        -------
        Candidate:
            The candidate to try on the objective function. Candidates have field "args" and "kwargs" which can be directly used
            on the function (objective_function(*candidate.args, **candidate.kwargs)).
        """
        # call callbacks for logging etc...
        for callback in self._callbacks.get("ask", []):
            callback(self)
        candidate = self._internal_ask_candidate()
        assert candidate is not None, f"{self.__class__.__name__}._internal_ask method returned None instead of a point."
        self._num_ask += 1
        if candidate.uuid in self._asked:
            raise RuntimeError("Cannot submit the same candidate twice: please recreate a new candidate from data.\n"
                               "This is to make sure that stochastic instrumentations are resampled.")
        self._asked.add(candidate.uuid)
        return candidate

    def provide_recommendation(self) -> Candidate:
        """Provides the best point to use as a minimum, given the budget that was used

        Returns
        -------
        Candidate
            The candidate with minimal value. Candidates have field "args" and "kwargs" which can be directly used
            on the function (objective_function(*candidate.args, **candidate.kwargs)).
        """
        return self.recommend()  # duplicate method

    def recommend(self) -> Candidate:
        """Provides the best candidate to use as a minimum, given the budget that was used.

        Returns
        -------
        Candidate
            The candidate with minimal value. Candidates have field "args" and "kwargs" which can be directly used
            on the function (objective_function(*candidate.args, **candidate.kwargs)).
        """
        return self.create_candidate.from_data(self._internal_provide_recommendation(), deterministic=True)

    def _internal_tell_not_asked(self, candidate: Candidate, value: float) -> None:
        """Called whenever calling "tell" on a candidate that was not "asked".
        Defaults to the standard tell pipeline.
        """
        self._internal_tell_candidate(candidate, value)

    def _internal_tell_candidate(self, candidate: Candidate, value: float) -> None:
        """Called whenever calling "tell" on a candidate that was "asked".
        """
        self._internal_tell(candidate.data, value)

    def _internal_ask_candidate(self) -> Candidate:
        return self.create_candidate.from_data(self._internal_ask())

    # Internal methods which can be overloaded (or must be, in the case of _internal_ask)
    def _internal_tell(self, x: ArrayLike, value: float) -> None:
        pass

    def _internal_ask(self) -> ArrayLike:
        raise RuntimeError("Not implemented, should not be called.")

    def _internal_provide_recommendation(self) -> ArrayLike:
        return self.current_bests["pessimistic"].x

    def optimize(self, objective_function: Callable[..., float],
                 executor: Optional[ExecutorLike] = None,
                 batch_mode: bool = False,
                 verbosity: int = 0) -> Candidate:
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

        Returns
        -------
        Candidate
            The candidate with minimal value. Candidates have field "args" and "kwargs" which can be directly used
            on the function (objective_function(*candidate.args, **candidate.kwargs)).

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
        assert executor is not None
        tmp_runnings: List[Tuple[Candidate, JobLike[float]]] = []
        tmp_finished: Deque[Tuple[Candidate, JobLike[float]]] = deque()
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
                new_sugg = max(0, min(remaining_budget, self.num_workers - len(self._running_jobs)))
                if verbosity and new_sugg:
                    print(f"Launching {new_sugg} jobs with new suggestions")
                for _ in range(new_sugg):
                    args = self.ask()
                    self._running_jobs.append((args, executor.submit(objective_function, *args.args, **args.kwargs)))
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
            print(f"After {optimizer.num_tell}, recommendation is {x}")  # TODO fetch value


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

    def __call__(self, instrumentation: Union[int, instru.Instrumentation],
                 budget: Optional[int] = None, num_workers: int = 1) -> Optimizer:
        raise NotImplementedError

    def load(self, filepath: Union[str, Path]) -> 'Optimizer':
        """Loads a pickle and checks that it is an Optimizer.
        """
        return load(Optimizer, filepath)


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

    def __call__(self, instrumentation: Union[int, instru.Instrumentation],
                 budget: Optional[int] = None, num_workers: int = 1) -> Optimizer:
        assert self._optimizer_class is not None
        run = self._optimizer_class(instrumentation=instrumentation, budget=budget, num_workers=num_workers)  # pylint: disable=not-callable
        assert hasattr(run, "_parameters")
        assert isinstance(run._parameters, self.__class__)  # type: ignore
        run._parameters = self  # type: ignore
        run.name = repr(self)
        return run

    def load(self, filepath: Union[str, Path]) -> 'Optimizer':
        """Loads a pickle and checks that it corresponds to the correct family of optimizer
        """
        assert self._optimizer_class is not None
        return load(self._optimizer_class, filepath)
