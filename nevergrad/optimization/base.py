# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import warnings
from pathlib import Path
from numbers import Real
from collections import deque
import numpy as np
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from nevergrad.common import tools as ngtools
from nevergrad.common import errors as errors
from nevergrad.common.decorators import Registry
from . import utils
from . import multiobjective as mobj


OptCls = tp.Union["ConfiguredOptimizer", tp.Type["Optimizer"]]
registry: Registry[OptCls] = Registry()
_OptimCallBack = tp.Union[
    tp.Callable[["Optimizer", "p.Parameter", float], None], tp.Callable[["Optimizer"], None]
]
X = tp.TypeVar("X", bound="Optimizer")
Y = tp.TypeVar("Y")
IntOrParameter = tp.Union[int, p.Parameter]
_PruningCallable = tp.Callable[[utils.Archive[utils.MultiValue]], utils.Archive[utils.MultiValue]]


def _loss(param: p.Parameter) -> float:
    """Returns the loss if available, or inf otherwise.
    Used to simplify handling of losses
    """
    return param.loss if param.loss is not None else float("inf")


def load(cls: tp.Type[X], filepath: tp.PathLike) -> X:
    """Loads a pickle file and checks that it contains an optimizer.
    The optimizer class is not always fully reliable though (e.g.: optimizer families) so the user is responsible for it.
    """
    filepath = Path(filepath)
    with filepath.open("rb") as f:
        opt = pickle.load(f)
    assert isinstance(opt, cls), f"You should only load {cls} with this method (found {type(opt)})"
    return opt


class Optimizer:  # pylint: disable=too-many-instance-attributes
    """Algorithm framework with 3 main functions:

    - :code:`ask()` which provides a candidate on which to evaluate the function to optimize.
    - :code:`tell(candidate, loss)` which lets you provide the loss associated to points.
    - :code:`provide_recommendation()` which provides the best final candidate.

    Typically, one would call :code:`ask()` num_workers times, evaluate the
    function on these num_workers points in parallel, update with the fitness value when the
    evaluations is finished, and iterate until the budget is over. At the very end,
    one would call provide_recommendation for the estimated optimum.

    This class is abstract, it provides internal equivalents for the 3 main functions,
    among which at least :code:`_internal_ask_candidate` has to be overridden.

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

    def __init__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> None:
        if self.no_parallelization and num_workers > 1:
            raise ValueError(f"{self.__class__.__name__} does not support parallelization")
        # "seedable" random state: externally setting the seed will provide deterministic behavior
        # you can also replace or reinitialize this random state
        self.num_workers = int(num_workers)
        self.budget = budget

        # How do we deal with cheap constraints i.e. constraints which are fast and use low resources and easy ?
        # True ==> we penalize them (infinite values for candidates which violate the constraint).
        # False ==> we repeat the ask until we solve the problem.
        self._constraints_manager = utils.ConstraintManager()
        self._penalize_cheap_violations = False

        self.parametrization = (
            parametrization
            if not isinstance(parametrization, (int, np.int))
            else p.Array(shape=(parametrization,))
        )
        self.parametrization.freeze()  # avoids issues!
        if not self.dimension:
            raise ValueError("No variable to optimize in this parametrization.")
        self.name = self.__class__.__name__  # printed name in repr
        # keep a record of evaluations, and current bests which are updated at each new evaluation
        self.archive: utils.Archive[
            utils.MultiValue
        ] = utils.Archive()  # dict like structure taking np.ndarray as keys and Value as values
        self.current_bests = {
            x: utils.MultiValue(self.parametrization, np.inf, reference=self.parametrization)
            for x in ["optimistic", "pessimistic", "average", "minimum"]
        }
        # pruning function, called at each "tell"
        # this can be desactivated or modified by each implementation
        self.pruning: tp.Optional[_PruningCallable] = utils.Pruning.sensible_default(
            num_workers=num_workers, dimension=self.parametrization.dimension
        )
        # multiobjective
        self._MULTIOBJECTIVE_AUTO_BOUND = mobj.AUTO_BOUND
        self._hypervolume_pareto: tp.Optional[mobj.HypervolumePareto] = None
        # instance state
        self._asked: tp.Set[str] = set()
        self._num_objectives = 0
        self._suggestions: tp.Deque[p.Parameter] = deque()
        self._num_ask = 0
        self._num_tell = 0  # increases after each successful tell
        self._num_tell_not_asked = 0
        self._callbacks: tp.Dict[str, tp.List[tp.Any]] = {}
        # to make optimize function stoppable halway through
        self._running_jobs: tp.List[tp.Tuple[p.Parameter, tp.JobLike[tp.Loss]]] = []
        self._finished_jobs: tp.Deque[tp.Tuple[p.Parameter, tp.JobLike[tp.Loss]]] = deque()

    @property
    def _rng(self) -> np.random.RandomState:
        """np.random.RandomState: parametrization random state the optimizer must pull from.
        It can be seeded or updated directly on the parametrization instance (`optimizer.parametrization.random_state`)
        """
        return self.parametrization.random_state

    @property
    def dimension(self) -> int:
        """int: Dimension of the optimization space."""
        return self.parametrization.dimension

    @property
    def num_objectives(self) -> int:
        """Provides 0 if the number is not known yet, else the number of objectives
        to optimize upon.
        """
        if (
            self._hypervolume_pareto is not None
            and self._num_objectives != self._hypervolume_pareto.num_objectives
        ):
            raise RuntimeError("Number of objectives is incorrectly set. Please create a nevergrad issue")
        return self._num_objectives

    @num_objectives.setter
    def num_objectives(self, num: int) -> None:
        num = int(num)
        if num <= 0:
            raise ValueError("Number of objectives must be strictly positive")
        if not self._num_objectives:
            self._num_objectives = num
            self._num_objectives_set_callback()
        elif num != self._num_objectives:
            raise ValueError(f"Expected {self._num_objectives} loss(es), but received {num}.")

    def _num_objectives_set_callback(self) -> None:
        """Callback for when num objectives is first known"""

    @property
    def num_ask(self) -> int:
        """int: Number of time the `ask` method was called."""
        return self._num_ask

    @property
    def num_tell(self) -> int:
        """int: Number of time the `tell` method was called."""
        return self._num_tell

    @property
    def num_tell_not_asked(self) -> int:
        """int: Number of time the :code:`tell` method was called on candidates that were not asked for by the optimizer
        (or were suggested).
        """
        return self._num_tell_not_asked

    def pareto_front(
        self, size: tp.Optional[int] = None, subset: str = "random", subset_tentatives: int = 12
    ) -> tp.List[p.Parameter]:
        """Pareto front, as a list of Parameter. The losses can be accessed through
        parameter.losses

        Parameters
        ------------
        size:  int (optional)
            if provided, selects a subset of the full pareto front with the given maximum size
        subset: str
            method for selecting the subset ("random, "loss-covering", "domain-covering", "hypervolume")
        subset_tentatives: int
            number of random tentatives for finding a better subset

        Returns
        --------
        list
            the list of Parameter of the pareto front

        Note
        ----
        During non-multiobjective optimization, this returns the current pessimistic best
        """
        pareto = (
            []
            if self._hypervolume_pareto is None
            else self._hypervolume_pareto.pareto_front(
                size=size, subset=subset, subset_tentatives=subset_tentatives
            )
        )
        return pareto if pareto else [self.provide_recommendation()]

    def dump(self, filepath: tp.Union[str, Path]) -> None:
        """Pickles the optimizer into a file."""
        filepath = Path(filepath)
        with filepath.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls: tp.Type[X], filepath: tp.Union[str, Path]) -> X:
        """Loads a pickle and checks that the class is correct."""
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
            name of the method to register the callback for (either :code:`ask` or :code:`tell`)
        callback: callable
            a callable taking the same parameters as the method it is registered upon (including self)
        """
        assert name in ["ask", "tell"], f'Only "ask" and "tell" methods can have callbacks (not {name})'
        self._callbacks.setdefault(name, []).append(callback)

    def remove_all_callbacks(self) -> None:
        """Removes all registered callables"""
        self._callbacks = {}

    def suggest(self, *args: tp.Any, **kwargs: tp.Any) -> None:
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
          Some optimizers may not support it and will raise a :code:`TellNotAskedNotSupportedError`
          at :code:`tell` time.
        - LIFO is used so as to be able to suggest and ask straightaway, as an alternative to
          creating a new candidate with :code:`optimizer.parametrization.spawn_child(new_value)`
        """
        if isinstance(self.parametrization, p.Instrumentation):
            new_value: tp.Any = (args, kwargs)
        else:
            assert len(args) == 1 and not kwargs
            new_value = args[0]
        self._suggestions.append(self.parametrization.spawn_child(new_value=new_value))

    # pylint: disable=too-many-branches
    def tell(self, candidate: p.Parameter, loss: tp.Loss) -> None:
        """Provides the optimizer with the evaluation of a fitness value for a candidate.

        Parameters
        ----------
        x: np.ndarray
            point where the function was evaluated
        loss: float/list/np.ndarray
            loss of the function (or multi-objective function

        Note
        ----
        The candidate should generally be one provided by :code:`ask()`, but can be also
        a non-asked candidate. To create a p.Parameter instance from args and kwargs,
        you can use :code:`candidate = optimizer.parametrization.spawn_child(new_value=your_value)`:

        - for an :code:`Array(shape(2,))`: :code:`optimizer.parametrization.spawn_child(new_value=[12, 12])`

        - for an :code:`Instrumentation`: :code:`optimizer.parametrization.spawn_child(new_value=(args, kwargs))`

        Alternatively, you can provide a suggestion with :code:`optimizer.suggest(*args, **kwargs)`, the next :code:`ask`
        will use this suggestion.
        """
        # Check loss type
        if isinstance(loss, (Real, float)):
            # using "float" along "Real" because mypy does not understand "Real" for now Issue #3186
            loss = float(loss)
            # Non-sense values including NaNs should not be accepted.
            # We do not use max-float as various later transformations could lead to greater values.
            if not loss < 5.0e20:  # pylint: disable=unneeded-not
                warnings.warn(
                    f"Clipping very high value {loss} in tell (rescale the cost function?).",
                    errors.LossTooLargeWarning,
                )
                loss = 5.0e20  # sys.float_info.max leads to numerical problems so let us do this.
        elif isinstance(loss, (tuple, list, np.ndarray)):
            loss = np.array(loss, copy=False, dtype=float).ravel() if len(loss) != 1 else loss[0]
        elif not isinstance(loss, np.ndarray):
            raise TypeError(
                f'"tell" method only supports float values but the passed loss was: {loss} (type: {type(loss)}.'
            )
        # check Parameter
        if not isinstance(candidate, p.Parameter):
            raise TypeError(
                "'tell' must be provided with the candidate.\n"
                "Use optimizer.parametrization.spawn_child(new_value)) if you want to "
                "create a candidate that as not been asked for, "
                "or optimizer.suggest(*args, **kwargs) to suggest a point that should be used for "
                "the next ask"
            )
        # check loss length
        self.num_objectives = 1 if isinstance(loss, float) else loss.size
        # checks are done, start processing
        candidate.freeze()  # make sure it is not modified somewhere
        # add reference if provided
        if isinstance(candidate, p.MultiobjectiveReference):
            if self._hypervolume_pareto is not None:
                raise RuntimeError("MultiobjectiveReference can only be provided before the first tell.")
            if not isinstance(loss, np.ndarray):
                raise RuntimeError("MultiobjectiveReference must only be used for multiobjective losses")
            self._hypervolume_pareto = mobj.HypervolumePareto(upper_bounds=loss, seed=self._rng)
            if candidate.value is None:
                return  # no value, so stopping processing there
            candidate = candidate.value
        # preprocess multiobjective loss
        if isinstance(loss, np.ndarray):
            candidate._losses = loss
        if not isinstance(loss, float):
            loss = self._preprocess_multiobjective(candidate)
        # call callbacks for logging etc...
        candidate.loss = loss
        assert isinstance(loss, float)
        for callback in self._callbacks.get("tell", []):
            # multiobjective reference is not handled :s
            # but this allows obtaining both scalar and multiobjective loss (through losses)
            callback(self, candidate, loss)
        if not candidate.satisfies_constraints() and self.budget is not None:
            penalty = self._constraints_manager.penalty(candidate, self.num_ask, self.budget)
            loss = loss + penalty
        if isinstance(loss, float):
            self._update_archive_and_bests(candidate, loss)
        if candidate.uid in self._asked:
            self._internal_tell_candidate(candidate, loss)
            self._asked.remove(candidate.uid)
        else:
            self._internal_tell_not_asked(candidate, loss)
            self._num_tell_not_asked += 1
        self._num_tell += 1

    def _preprocess_multiobjective(self, candidate: p.Parameter) -> tp.FloatLoss:
        if self._hypervolume_pareto is None:
            self._hypervolume_pareto = mobj.HypervolumePareto(auto_bound=self._MULTIOBJECTIVE_AUTO_BOUND)
        return self._hypervolume_pareto.add(candidate)

    def _update_archive_and_bests(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        x = candidate.get_standardized_data(reference=self.parametrization)
        if not isinstance(
            loss, (Real, float)
        ):  # using "float" along "Real" because mypy does not understand "Real" for now Issue #3186
            raise TypeError(
                f'"tell" method only supports float values but the passed loss was: {loss} (type: {type(loss)}.'
            )
        if np.isnan(loss) or loss == np.inf:
            warnings.warn(f"Updating fitness with {loss} value", errors.BadLossWarning)
        mvalue: tp.Optional[utils.MultiValue] = None
        if x not in self.archive:
            self.archive[x] = utils.MultiValue(candidate, loss, reference=self.parametrization)
        else:
            mvalue = self.archive[x]
            mvalue.add_evaluation(loss)
            # both parameters should be non-None
            if mvalue.parameter.loss > candidate.loss:  # type: ignore
                mvalue.parameter = candidate  # keep best candidate
        # update current best records
        # this may have to be improved if we want to keep more kinds of best losss

        for name in self.current_bests:
            if mvalue is self.current_bests[name]:  # reboot
                best = min(self.archive.values(), key=lambda mv, n=name: mv.get_estimation(n))  # type: ignore
                # rebuild best point may change, and which value did not track the updated value anyway
                self.current_bests[name] = best
            else:
                if self.archive[x].get_estimation(name) <= self.current_bests[name].get_estimation(name):
                    self.current_bests[name] = self.archive[x]
                # deactivated checks
                # if not (np.isnan(loss) or loss == np.inf):
                #     if not self.current_bests[name].x in self.archive:
                #         bval = self.current_bests[name].get_estimation(name)
                #         avals = (min(v.get_estimation(name) for v in self.archive.values()),
                #                  max(v.get_estimation(name) for v in self.archive.values()))
                #         raise RuntimeError(f"Best value should exist in the archive at num_tell={self.num_tell})\n"
                #                            f"Best value is {bval} and archive is within range {avals} for {name}")
        if self.pruning is not None:
            self.archive = self.pruning(self.archive)

    def ask(self) -> p.Parameter:
        """Provides a point to explore.
        This function can be called multiple times to explore several points in parallel

        Returns
        -------
        p.Parameter:
            The candidate to try on the objective function. :code:`p.Parameter` have field :code:`args` and :code:`kwargs`
            which can be directly used on the function (:code:`objective_function(*candidate.args, **candidate.kwargs)`).
        """
        # call callbacks for logging etc...
        for callback in self._callbacks.get("ask", []):
            callback(self)
        current_num_ask = self.num_ask
        # tentatives if a cheap constraint is available
        max_trials = max(1, self._constraints_manager.max_trials // 2)
        # half will be used for sub-optimization --- if the optimization method does not need/use a budget.
        # TODO(oteytaud): actually we could do this even when the budget is known, if we are sure that
        # exceeding the budget is not a problem.
        # Very simple constraint solver:
        # - we use a simple algorithm.
        # - no memory of previous iterations.
        # - just projection to constraint satisfaction.
        # We try using the normal tool during half constraint budget, in order to reduce the impact on the normal run.
        for _ in range(max_trials):
            is_suggestion = False
            if self._suggestions:  # use suggestions if available
                is_suggestion = True
                candidate = self._suggestions.pop()
            else:
                candidate = self._internal_ask_candidate()
            if candidate.satisfies_constraints():
                break  # good to go!
            if self._penalize_cheap_violations:
                # Warning! This might be a tell not asked.
                self._internal_tell_candidate(candidate, float("Inf"))  # DE requires a tell
            # updating num_ask  is necessary for some algorithms which need new num to ask another point
            self._num_ask += 1
        satisfies = candidate.satisfies_constraints()
        if not satisfies:
            # still not solving, let's run sub-optimization
            candidate = _constraint_solver(candidate, budget=max_trials)
        if not (satisfies or candidate.satisfies_constraints()):
            warnings.warn(
                f"Could not bypass the constraint after {max_trials} tentatives, "
                "sending candidate anyway.",
                errors.FailedConstraintWarning,
            )
        if not is_suggestion:
            if candidate.uid in self._asked:
                raise RuntimeError(
                    "Cannot submit the same candidate twice: please recreate a new candidate from data.\n"
                    "This is to make sure that stochastic parametrizations are resampled."
                )
            self._asked.add(candidate.uid)
        self._num_ask = current_num_ask + 1
        assert (
            candidate is not None
        ), f"{self.__class__.__name__}._internal_ask method returned None instead of a point."
        # make sure to call value getter which may update the value, before we freeze the paremeter
        candidate.value  # pylint: disable=pointless-statement
        candidate.freeze()  # make sure it is not modified somewhere
        return candidate

    def provide_recommendation(self) -> p.Parameter:
        """Provides the best point to use as a minimum, given the budget that was used

        Returns
        -------
        p.Parameter
            The candidate with minimal value. p.Parameters have field :code:`args` and :code:`kwargs` which can be directly used
            on the function (:code:`objective_function(*candidate.args, **candidate.kwargs)`).
        """
        return self.recommend()  # duplicate method

    def recommend(self) -> p.Parameter:
        """Provides the best candidate to use as a minimum, given the budget that was used.

        Returns
        -------
        p.Parameter
            The candidate with minimal loss. :code:`p.Parameters` have field :code:`args` and :code:`kwargs` which can be directly used
            on the function (:code:`objective_function(*candidate.args, **candidate.kwargs)`).
        """
        recom_data = self._internal_provide_recommendation()  # pylint: disable=assignment-from-none
        if recom_data is None or any(np.isnan(recom_data)):
            name = "minimum" if self.parametrization.function.deterministic else "pessimistic"
            return self.current_bests[name].parameter
        out = self.parametrization.spawn_child()
        with p.helpers.deterministic_sampling(out):
            out.set_standardized_data(recom_data)
        return out

    def _internal_tell_not_asked(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        """Called whenever calling :code:`tell` on a candidate that was not "asked".
        Defaults to the standard tell pipeline.
        """
        self._internal_tell_candidate(candidate, loss)

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        """Called whenever calling :code:`tell` on a candidate that was "asked"."""
        data = candidate.get_standardized_data(reference=self.parametrization)
        self._internal_tell(data, loss)

    def _internal_ask_candidate(self) -> p.Parameter:
        return self.parametrization.spawn_child().set_standardized_data(self._internal_ask())

    # Internal methods which can be overloaded (or must be, in the case of _internal_ask)
    def _internal_tell(self, x: tp.ArrayLike, loss: tp.FloatLoss) -> None:
        pass

    def _internal_ask(self) -> tp.ArrayLike:
        raise RuntimeError("Not implemented, should not be called.")

    def _internal_provide_recommendation(self) -> tp.Optional[tp.ArrayLike]:
        """Override to provide a recommendation in standardized space"""
        return None

    def minimize(
        self,
        objective_function: tp.Callable[..., tp.Loss],
        executor: tp.Optional[tp.ExecutorLike] = None,
        batch_mode: bool = False,
        verbosity: int = 0,
    ) -> p.Parameter:
        """Optimization (minimization) procedure

        Parameters
        ----------
        objective_function: callable
            A callable to optimize (minimize)
        executor: Executor
            An executor object, with method :code:`submit(callable, *args, **kwargs)` and returning a Future-like object
            with methods :code:`done() -> bool` and :code:`result() -> float`. The executor role is to dispatch the execution of
            the jobs locally/on a cluster/with multithreading depending on the implementation.
            Eg: :code:`concurrent.futures.ThreadPoolExecutor`
        batch_mode: bool
            when :code:`num_workers = n > 1`, whether jobs are executed by batch (:code:`n` function evaluations are launched,
            we wait for all results and relaunch n evals) or not (whenever an evaluation is finished, we launch
            another one)
        verbosity: int
            print information about the optimization (0: None, 1: fitness values, 2: fitness values and recommendation)

        Returns
        -------
        ng.p.Parameter
            The candidate with minimal value. :code:`ng.p.Parameters` have field :code:`args` and :code:`kwargs` which can
            be directly used on the function (:code:`objective_function(*candidate.args, **candidate.kwargs)`).

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
                warnings.warn(
                    f"num_workers = {self.num_workers} > 1 is suboptimal when run sequentially",
                    errors.InefficientSettingsWarning,
                )
        assert executor is not None
        tmp_runnings: tp.List[tp.Tuple[p.Parameter, tp.JobLike[tp.Loss]]] = []
        tmp_finished: tp.Deque[tp.Tuple[p.Parameter, tp.JobLike[tp.Loss]]] = deque()
        # go
        sleeper = ngtools.Sleeper()  # manages waiting time depending on execution time of the jobs
        remaining_budget = self.budget - self.num_ask
        first_iteration = True
        #
        while remaining_budget or self._running_jobs or self._finished_jobs:
            # # # # # Update optimizer with finished jobs # # # # #
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
            # # # # # Start new jobs # # # # #
            if not batch_mode or not self._running_jobs:
                new_sugg = max(0, min(remaining_budget, self.num_workers - len(self._running_jobs)))
                if verbosity and new_sugg:
                    print(f"Launching {new_sugg} jobs with new suggestions")
                for _ in range(new_sugg):
                    try:
                        args = self.ask()
                    except errors.NevergradEarlyStopping:
                        remaining_budget = 0
                        break
                    self._running_jobs.append(
                        (args, executor.submit(objective_function, *args.args, **args.kwargs))
                    )
                if new_sugg:
                    sleeper.start_timer()
            if remaining_budget > 0:  # early stopping sets it to 0
                remaining_budget = self.budget - self.num_ask
            # split (repopulate finished and runnings in only one loop to avoid
            # weird effects if job finishes in between two list comprehensions)
            tmp_runnings, tmp_finished = [], deque()
            for x_job in self._running_jobs:
                (tmp_finished if x_job[1].done() else tmp_runnings).append(x_job)
            self._running_jobs, self._finished_jobs = tmp_runnings, tmp_finished
            first_iteration = False
        return self.provide_recommendation()


# Adding a comparison-only functionality to an optimizer.
def addCompare(optimizer: Optimizer) -> None:
    def compare(self: Optimizer, winners: tp.List[p.Parameter], losers: tp.List[p.Parameter]) -> None:
        # This means that for any i and j, winners[i] is better than winners[i+1], and better than losers[j].
        # This is for cases in which we do not know fitness values, we just know comparisons.

        ref = self.parametrization
        # Evaluate the best fitness value among losers.
        best_fitness_value = 0.0
        for candidate in losers:
            data = candidate.get_standardized_data(reference=self.parametrization)
            if data in self.archive:
                best_fitness_value = min(best_fitness_value, self.archive[data].get_estimation("average"))

        # Now let us decide the fitness value of winners.
        for i, candidate in enumerate(winners):
            self.tell(candidate, best_fitness_value - len(winners) + i)
            data = candidate.get_standardized_data(reference=self.parametrization)
            self.archive[data] = utils.MultiValue(
                candidate, best_fitness_value - len(winners) + i, reference=ref
            )

    setattr(optimizer.__class__, "compare", compare)


class ConfiguredOptimizer:
    """Creates optimizer-like instances with configuration.

    Parameters
    ----------
    OptimizerClass: type
        class of the optimizer to configure
    config: dict
        dictionnary of all the configurations
    as_config: bool
        whether to provide all config as kwargs to the optimizer instantiation (default, see ConfiguredCMA for an example),
        or through a config kwarg referencing self. (if True, see EvolutionStrategy for an example)

    Note
    ----
    This provides a default repr which can be bypassed through set_name
    """

    # optimizer qualifiers
    recast = False  # algorithm which were not designed to work with the suggest/update pattern
    one_shot = False  # algorithm designed to suggest all budget points at once
    no_parallelization = False  # algorithm which is designed to run sequentially only

    def __init__(
        self, OptimizerClass: tp.Type[Optimizer], config: tp.Dict[str, tp.Any], as_config: bool = False
    ) -> None:
        self._OptimizerClass = OptimizerClass
        config.pop("self", None)  # self comes from "locals()"
        config.pop("__class__", None)  # self comes from "locals()"
        self._as_config = as_config
        self._config = config  # keep all, to avoid weird behavior at mismatch between optim and configoptim
        diff = ngtools.different_from_defaults(instance=self, instance_dict=config, check_mismatches=True)
        params = ", ".join(f"{x}={y!r}" for x, y in sorted(diff.items()))
        self.name = f"{self.__class__.__name__}({params})"
        if not as_config:
            # try instantiating for init checks
            # if as_config: check can be done before setting attributes
            self(parametrization=4, budget=100)

    def config(self) -> tp.Dict[str, tp.Any]:
        return dict(self._config)

    def __call__(
        self, parametrization: IntOrParameter, budget: tp.Optional[int] = None, num_workers: int = 1
    ) -> Optimizer:
        """Creates an optimizer from the parametrization

        Parameters
        ----------
        instrumentation: int or Instrumentation
            either the dimension of the optimization space, or its instrumentation
        budget: int/None
            number of allowed evaluations
        num_workers: int
            number of evaluations which will be run in parallel at once
        """
        config = dict(config=self) if self._as_config else self._config
        run = self._OptimizerClass(parametrization=parametrization, budget=budget, num_workers=num_workers, **config)  # type: ignore
        run.name = self.name
        # hacky but convenient to have around:
        run._configured_optimizer = self  # type: ignore
        return run

    def __repr__(self) -> str:
        return self.name

    def set_name(self, name: str, register: bool = False) -> "ConfiguredOptimizer":
        """Set a new representation for the instance"""
        self.name = name
        if register:
            registry.register_name(name, self)
        return self

    def load(self, filepath: tp.Union[str, Path]) -> "Optimizer":
        """Loads a pickle and checks that it is an Optimizer."""
        return self._OptimizerClass.load(filepath)

    def __eq__(self, other: tp.Any) -> tp.Any:
        if self.__class__ == other.__class__:
            if self._config == other._config:
                return True
        return False


def _constraint_solver(parameter: p.Parameter, budget: int) -> p.Parameter:
    """Runs a suboptimization to solve the parameter constraints"""
    parameter_without_constraint = parameter.copy()
    parameter_without_constraint._constraint_checkers.clear()
    opt = registry["OnePlusOne"](parameter_without_constraint, num_workers=1, budget=budget)
    for _ in range(budget):
        cand = opt.ask()
        # Our objective function is minimum for the point the closest to
        # the original candidate under the constraints.
        penalty = sum(utils._float_penalty(func(cand.value)) for func in parameter._constraint_checkers)

        # TODO: this may not scale well with dimension
        distance = np.tanh(np.sum(cand.get_standardized_data(reference=parameter) ** 2))
        # TODO: because of the return whenever constraints are satisfied, the first case never arises
        loss = distance if penalty <= 0 else penalty + distance + 1.0
        opt.tell(cand, loss)
        if penalty <= 0:  # constraints are satisfied
            break
    data = opt.recommend().get_standardized_data(reference=parameter_without_constraint)
    return parameter.spawn_child().set_standardized_data(data)
