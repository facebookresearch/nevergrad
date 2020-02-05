# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import pickle
import warnings
from pathlib import Path
from numbers import Real
from collections import deque
import typing as tp  # favor using tp.Dict instead of Dict etc
from typing import Optional, Tuple, Callable, Any, Dict, List, Union, Deque, Type, Set, TypeVar
import numpy as np
from nevergrad.parametrization import parameter as p
from nevergrad.common import tools as ngtools
from ..common.typetools import ArrayLike as ArrayLike  # allows reexport
from ..common.typetools import JobLike, ExecutorLike
from ..common.decorators import Registry
from . import utils


registry = Registry[Union["OptimizerFamily", Type["Optimizer"]]]()
_OptimCallBack = Union[Callable[["Optimizer", "p.Parameter", float], None], Callable[["Optimizer"], None]]
X = TypeVar("X", bound="Optimizer")
Y = TypeVar("Y")
IntOrParameter = tp.Union[int, p.Parameter]


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


class CandidateMaker:

    def from_call(self, *args: Any, **kwargs: Any) -> p.Parameter:
        raise RuntimeError("CandidateMaker is deprecated, use parametrization.spawn_child(new_value=(args, kwargs)) instead")

    def from_data(self, data: ArrayLike, deterministic: bool = False) -> p.Parameter:
        raise RuntimeError("CandidateMaker is deprecated, "
                           "use parametrization.spawn_child().set_standardized_data(data, deterministic) instead")


def deprecated_init(func: tp.Callable[..., Y]) -> tp.Callable[..., Y]:

    def _deprecated_init_wrapper(
        self: "Optimizer",
        parametrization: tp.Optional[IntOrParameter] = None,
        budget: Optional[int] = None,
        num_workers: int = 1,
        instrumentation: tp.Optional[IntOrParameter] = None,
        **kwargs: tp.Any,
    ) -> Y:
        assert func.__name__ in ["__call__", "__init__"]
        assert func is not None
        if instrumentation is not None:
            warnings.warn('"instrumentation" __init__ parameter has been renamed to "parametrization" for consistency. '
                          "using it will not be supported starting at v0.4.0 (coming soon!)", DeprecationWarning)
            if parametrization is not None:
                raise ValueError('Only parametrization arguement should be specified, not "instrumentation" which is deprecated')
            parametrization = instrumentation
        assert parametrization is not None, '"parametrization" must be provided to the optimizer'
        assert isinstance(parametrization, (int, p.Parameter)), f"Weird input {parametrization}"
        return func(self, parametrization, budget, num_workers, **kwargs)

    return _deprecated_init_wrapper


class Optimizer:  # pylint: disable=too-many-instance-attributes
    """Algorithm framework with 3 main functions:

    - `ask()` which provides a candidate on which to evaluate the function to optimize.
    - `tell(candidate, value)` which lets you provide the values associated to points.
    - `provide_recommendation()` which provides the best final candidate.

    Typically, one would call `ask()` num_workers times, evaluate the
    function on these num_workers points in parallel, update with the fitness value when the
    evaluations is finished, and iterate until the budget is over. At the very end,
    one would call provide_recommendation for the estimated optimum.

    This class is abstract, it provides internal equivalents for the 3 main functions,
    among which at least `_internal_ask_candidate` has to be overridden.

    Each optimizer instance should be used only once, with the initial provided budget

    Parameters
    ----------
    parametrization: int or Parameter
        either the dimension of the optimization space, or its parametrization
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

    @deprecated_init
    def __init__(self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1) -> None:
        if self.no_parallelization and num_workers > 1:
            raise ValueError(f"{self.__class__.__name__} does not support parallelization")
        # "seedable" random state: externally setting the seed will provide deterministic behavior
        # you can also replace or reinitialize this random state
        self.num_workers = int(num_workers)
        self.budget = budget
        # How do we deal with cheap constraints i.e. constraints which are fast and use low resources and easy ?
        # True ==> we penalize them (infinite values for candidates which violate the constraint).
        # False ==> we repeat the ask until we solve the problem.
        self._penalize_cheap_violations = False
        self.parametrization = (
            parametrization
            if not isinstance(parametrization, (int, np.int))
            else p.Array(shape=(parametrization,))
        )
        self.parametrization.freeze()  # avoids issues!
        if not self.dimension:
            raise ValueError("No variable to optimize in this parametrization.")
        self.create_candidate = CandidateMaker()
        self.name = self.__class__.__name__  # printed name in repr
        # keep a record of evaluations, and current bests which are updated at each new evaluation
        self.archive: utils.Archive[utils.Value] = utils.Archive()  # dict like structure taking np.ndarray as keys and Value as values
        self.current_bests = {
            x: utils.Point(np.zeros(self.dimension, dtype=np.float), utils.Value(np.inf)) for x in ["optimistic", "pessimistic", "average"]
        }
        # pruning function, called at each "tell"
        # this can be desactivated or modified by each implementation
        self.pruning: Optional[Callable[[utils.Archive[utils.Value]], utils.Archive[utils.Value]]] = utils.Pruning.sensible_default(
            num_workers=num_workers, dimension=self.parametrization.dimension
        )
        # instance state
        self._asked: Set[str] = set()
        self._suggestions: Deque[p.Parameter] = deque()
        self._num_ask = 0
        self._num_tell = 0
        self._num_tell_not_asked = 0
        self._callbacks: Dict[str, List[Any]] = {}
        # to make optimize function stoppable halway through
        self._running_jobs: List[Tuple[p.Parameter, JobLike[float]]] = []
        self._finished_jobs: Deque[Tuple[p.Parameter, JobLike[float]]] = deque()

    @property
    def instrumentation(self) -> p.Parameter:
        warnings.warn('"instrumentation" attribute has been renamed to "parametrization" for consistency. '
                      "using it will not be supported starting at v0.4.0 (coming soon!)", DeprecationWarning)
        return self.parametrization

    @property
    def _rng(self) -> np.random.RandomState:
        """np.random.RandomState: parametrization random state the optimizer must pull from.
        It can be seeded or updated directly on the parametrization instance (`optimizer.parametrization.random_state`)
        """
        return self.parametrization.random_state

    @property
    def dimension(self) -> int:
        """int: Dimension of the optimization space.
        """
        return self.parametrization.dimension

    @property
    def num_ask(self) -> int:
        """int: Number of time the `ask` method was called.
        """
        return self._num_ask

    @property
    def num_tell(self) -> int:
        """int: Number of time the `tell` method was called.
        """
        return self._num_tell

    @property
    def num_tell_not_asked(self) -> int:
        """int: Number of time the `tell` method was called on candidates that were not asked for by the optimizer
        (or were suggested).
        """
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
        inststr = self.parametrization.name
        return f"Instance of {self.name}(parametrization={inststr}, budget={self.budget}, num_workers={self.num_workers})"

    def register_callback(self, name: str, callback: _OptimCallBack) -> None:
        """Add a callback method called either when `tell` or `ask` are called, with the same
        arguments (including the optimizer / self). This can be useful for custom logging.

        Parameters
        ----------
        name: str
            name of the method to register the callback for (either `ask` or `tell`)
        callback: callable
            a callable taking the same parameters as the method it is registered upon (including self)
        """
        assert name in ["ask", "tell"], f'Only "ask" and "tell" methods can have callbacks (not {name})'
        self._callbacks.setdefault(name, []).append(callback)

    def remove_all_callbacks(self) -> None:
        """Removes all registered callables
        """
        self._callbacks = {}

    def suggest(self, *args: Any, **kwargs: Any) -> None:
        """Suggests a new point to ask.
        It will be asked at the next call (last in first out).

        Parameters
        ----------
        *args: Any
            positional arguments matching the parametrization pattern.
        *kwargs: Any
            keyword arguments matching the parametrization pattern.

        Note
        ----
        - This relies on optmizers implementing a way to deal with unasked candidate.
          Some optimizers may not support it and will raise a TellNotAskedNotSupportedError
          at "tell" time.
        - LIFO is used so as to be able to suggest and ask straightaway, as an alternative to
          calling optimizer.create_candidate.from_call.
        """
        if isinstance(self.parametrization, p.Instrumentation):
            new_value: tp.Any = (args, kwargs)
        else:
            assert len(args) == 1 and not kwargs
            new_value = args[0]
        self._suggestions.append(self.parametrization.spawn_child(new_value=new_value))

    def tell(self, candidate: p.Parameter, value: float) -> None:
        """Provides the optimizer with the evaluation of a fitness value for a candidate.

        Parameters
        ----------
        x: np.ndarray
            point where the function was evaluated
        value: float
            value of the function

        Note
        ----
        The candidate should generally be one provided by `ask()`, but can be also
        a non-asked candidate. To create a p.Parameter instance from args and kwargs,
        you can use `optimizer.create_candidate.from_call(*args, **kwargs)`.
        """
        if not isinstance(candidate, p.Parameter):
            raise TypeError(
                "'tell' must be provided with the candidate (use optimizer.create_candidate.from_call(*args, **kwargs)) "
                "if you want to inoculate a point that as not been asked for"
            )
        candidate.freeze()  # make sure it is not modified somewhere
        # call callbacks for logging etc...
        for callback in self._callbacks.get("tell", []):
            callback(self, candidate, value)
        data = candidate.get_standardized_data(reference=self.parametrization)
        self._update_archive_and_bests(data, value)
        if candidate.uid in self._asked:
            self._internal_tell_candidate(candidate, value)
            self._asked.remove(candidate.uid)
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
            if np.array_equal(x, self.current_bests[name].x):  # reboot
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

    def ask(self) -> p.Parameter:
        """Provides a point to explore.
        This function can be called multiple times to explore several points in parallel

        Returns
        -------
        p.Parameter:
            The candidate to try on the objective function. p.Parameter have field `args` and `kwargs` which can be directly used
            on the function (`objective_function(*candidate.args, **candidate.kwargs)`).
        """
        # call callbacks for logging etc...
        for callback in self._callbacks.get("ask", []):
            callback(self)
        current_num_ask = self.num_ask
        # tentatives if a cheap constraint is available
        MAX_TENTATIVES = 1000
        for k in range(MAX_TENTATIVES):
            is_suggestion = False
            if self._suggestions:
                is_suggestion = True
                candidate = self._suggestions.pop()
            else:
                candidate = self._internal_ask_candidate()
                # only register actual asked points
            if candidate.satisfies_constraints():
                break  # good to go!
            else:
                if self._penalize_cheap_violations or k == MAX_TENTATIVES - 2:  # a tell may help before last tentative
                    self._internal_tell_candidate(candidate, float("Inf"))
                self._num_ask += 1  # this is necessary for some algorithms which need new num to ask another point
                if k == MAX_TENTATIVES - 1:
                    warnings.warn(f"Could not bypass the constraint after {MAX_TENTATIVES} tentatives, sending candidate anyway.")
        if not is_suggestion:
            if candidate.uid in self._asked:
                raise RuntimeError(
                    "Cannot submit the same candidate twice: please recreate a new candidate from data.\n"
                    "This is to make sure that stochastic parametrizations are resampled."
                )
            self._asked.add(candidate.uid)
        self._num_ask = current_num_ask + 1
        assert candidate is not None, f"{self.__class__.__name__}._internal_ask method returned None instead of a point."
        candidate.freeze()  # make sure it is not modified somewhere
        return candidate

    def provide_recommendation(self) -> p.Parameter:
        """Provides the best point to use as a minimum, given the budget that was used

        Returns
        -------
        p.Parameter
            The candidate with minimal value. p.Parameters have field `args` and `kwargs` which can be directly used
            on the function (`objective_function(*candidate.args, **candidate.kwargs)`).
        """
        return self.recommend()  # duplicate method

    def recommend(self) -> p.Parameter:
        """Provides the best candidate to use as a minimum, given the budget that was used.

        Returns
        -------
        p.Parameter
            The candidate with minimal value. p.Parameters have field `args` and `kwargs` which can be directly used
            on the function (`objective_function(*candidate.args, **candidate.kwargs)`).
        """
        return self.parametrization.spawn_child().set_standardized_data(self._internal_provide_recommendation(), deterministic=True)

    def _internal_tell_not_asked(self, candidate: p.Parameter, value: float) -> None:
        """Called whenever calling "tell" on a candidate that was not "asked".
        Defaults to the standard tell pipeline.
        """
        self._internal_tell_candidate(candidate, value)

    def _internal_tell_candidate(self, candidate: p.Parameter, value: float) -> None:
        """Called whenever calling "tell" on a candidate that was "asked".
        """
        data = candidate.get_standardized_data(reference=self.parametrization)
        self._internal_tell(data, value)

    def _internal_ask_candidate(self) -> p.Parameter:
        return self.parametrization.spawn_child().set_standardized_data(self._internal_ask())

    # Internal methods which can be overloaded (or must be, in the case of _internal_ask)
    def _internal_tell(self, x: ArrayLike, value: float) -> None:
        pass

    def _internal_ask(self) -> ArrayLike:
        raise RuntimeError("Not implemented, should not be called.")

    def _internal_provide_recommendation(self) -> ArrayLike:
        return self.current_bests["pessimistic"].x

    def minimize(
        self,
        objective_function: Callable[..., float],
        executor: Optional[ExecutorLike] = None,
        batch_mode: bool = False,
        verbosity: int = 0,
    ) -> p.Parameter:
        """Optimization (minimization) procedure

        Parameters
        ----------
        objective_function: callable
            A callable to optimize (minimize)
        executor: Executor
            An executor object, with method `submit(callable, *args, **kwargs)` and returning a Future-like object
            with methods `done() -> bool` and `result() -> float`. The executor role is to dispatch the execution of
            the jobs locally/on a cluster/with multithreading depending on the implementation.
            Eg: `concurrent.futures.ThreadPoolExecutor`
        batch_mode: bool
            when num_workers = n > 1, whether jobs are executed by batch (n function evaluations are launched,
            we wait for all results and relaunch n evals) or not (whenever an evaluation is finished, we launch
            another one)
        verbosity: int
            print information about the optimization (0: None, 1: fitness values, 2: fitness values and recommendation)

        Returns
        -------
        p.Parameter
            The candidate with minimal value. p.Parameters have field `args` and `kwargs` which can be directly used
            on the function (`objective_function(*candidate.args, **candidate.kwargs)`).

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
        tmp_runnings: List[Tuple[p.Parameter, JobLike[float]]] = []
        tmp_finished: Deque[Tuple[p.Parameter, JobLike[float]]] = deque()
        # go
        sleeper = ngtools.Sleeper()  # manages waiting time depending on execution time of the jobs
        remaining_budget = self.budget - self.num_ask
        first_iteration = True
        # multiobjective hack
        func = objective_function
        multiobjective = hasattr(func, "multiobjective_function")
        if multiobjective:
            func = func.multiobjective_function  # type: ignore
        #
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
                    result = job.result()
                    if multiobjective:  # hack
                        result = objective_function.compute_aggregate_loss(job.result(), *x.args, **x.kwargs)  # type: ignore
                    self.tell(x, result)
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
                    self._running_jobs.append((args, executor.submit(func, *args.args, **args.kwargs)))
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

    def optimize(
        self,
        objective_function: Callable[..., float],
        executor: Optional[ExecutorLike] = None,
        batch_mode: bool = False,
        verbosity: int = 0,
    ) -> p.Parameter:
        """This function is deprecated and renamed "minimize".
        """
        warnings.warn("'optimize' method is deprecated, please use 'minimize' for clarity", DeprecationWarning)
        return self.minimize(objective_function, executor=executor, batch_mode=batch_mode, verbosity=verbosity)


# Adding a comparison-only functionality to an optimizer.
def addCompare(optimizer: Optimizer) -> None:

    def compare(self: Optimizer, winners: List[p.Parameter], losers: List[p.Parameter]) -> None:
        # This means that for any i and j, winners[i] is better than winners[i+1], and better than losers[j].
        # This is for cases in which we do not know fitness values, we just know comparisons.

        # Evaluate the best fitness value among losers.
        best_fitness_value = 0.
        for candidate in losers:
            data = candidate.get_standardized_data(reference=self.parametrization)
            if data in self.archive:
                best_fitness_value = min(best_fitness_value, self.archive[data].get_estimation("average"))

        # Now let us decide the fitness value of winners.
        for i, candidate in enumerate(winners):
            self.tell(candidate, best_fitness_value - len(winners) + i)
            data = candidate.get_standardized_data(reference=self.parametrization)
            self.archive[data] = utils.Value(best_fitness_value - len(winners) + i)

    setattr(optimizer.__class__, 'compare', compare)


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

    def with_name(self, name: str, register: bool = False) -> "OptimizerFamily":
        self._repr = name
        if register:
            registry.register_name(name, self)
        return self

    @deprecated_init
    def __call__(
        self, parametrization: IntOrParameter, budget: Optional[int] = None, num_workers: int = 1
    ) -> Optimizer:
        raise NotImplementedError

    def load(self, filepath: Union[str, Path]) -> "Optimizer":
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
        different = ngtools.different_from_defaults(self, check_mismatches=True)
        super().__init__(**different)

    def config(self) -> tp.Dict[str, tp.Any]:
        return {x: y for x, y in self.__dict__.items() if not x.startswith("_")}

    @deprecated_init
    def __call__(
        self, parametrization: tp.Optional[IntOrParameter] = None, budget: Optional[int] = None, num_workers: int = 1
    ) -> Optimizer:
        """Creates an optimizer from the parametrization

        Parameters
        ----------
        parametrization: int or Parameter
            either the dimension of the optimization space, or its parametrization
        budget: int/None
            number of allowed evaluations
        num_workers: int
            number of evaluations which will be run in parallel at once
        """
        assert self._optimizer_class is not None
        # pylint: disable=not-callable
        run = self._optimizer_class(parametrization, budget, num_workers)
        assert hasattr(run, "_parameters")
        assert isinstance(run._parameters, self.__class__)  # type: ignore
        run._parameters = self  # type: ignore
        run.name = repr(self)
        return run

    def load(self, filepath: Union[str, Path]) -> "Optimizer":
        """Loads a pickle and checks that it corresponds to the correct family of optimizer
        """
        assert self._optimizer_class is not None
        return load(self._optimizer_class, filepath)
