# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time
import random
import warnings
import traceback
from typing import Dict, Union, Callable, Any, Optional, Iterator, Tuple
import numpy as np
from ..common import decorators
from ..functions import BaseFunction
from ..optimization import base
from ..optimization.optimizerlib import registry as optimizer_registry
from . import execution


registry = decorators.Registry()


class CallCounter(execution.PostponedObject):
    """Simple wrapper which counts the number
    of calls to a function.

    Parameter
    ---------
    func: Callable
        the callable to wrap
    """

    def __init__(self, func: Callable) -> None:
        self.func = func
        self.num_calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.num_calls += 1
        return self.func(*args, **kwargs)

    def get_postponing_delay(self, arguments: Tuple[Tuple[Any, ...], Dict[str, Any]], value: float) -> float:
        """Propagate subfunction delay
        """
        if isinstance(self.func, execution.PostponedObject):
            return self.func.get_postponing_delay(arguments, value)
        return 0


class OptimizerSettings:
    """Handle for optimizer settings (name, num_workers etc)
    Optimizers can be instanciated through this class, providing the optimization space dimension.

    Note
    ----
    Eventually, this class should be moved to be directly used for defining experiments.
    """

    def __init__(self, name: str, budget: int, num_workers: int = 1, batch_mode: bool = True) -> None:
        self._setting_names = [x for x in locals() if x != "self"]
        assert name in optimizer_registry, f"{name} is not registered"
        self.name = name
        self.budget = budget
        self.num_workers = num_workers
        self.batch_mode = batch_mode

    def __repr__(self) -> str:
        return f"Experiment: {self.name}<budget={self.budget}, num_workers={self.num_workers}, batch_mode={self.batch_mode}>"

    @property
    def is_incoherent(self) -> bool:
        """Flags settings which are known to be impossible to process.
        Currently, this means we flag:
        - no_parallelization optimizers for num_workers > 1
        """
        # flag no_parallelization when num_workers greater than 1
        optimizer = optimizer_registry[self.name]
        return optimizer.no_parallelization and bool(self.num_workers > 1)

    def instanciate(self, dimension: int) -> base.Optimizer:
        """Instanciate an optimizer, providing the optimization space dimension
        """
        optim: base.Optimizer = optimizer_registry[self.name](dimension=dimension, budget=self.budget, num_workers=self.num_workers)
        return optim

    def get_description(self) -> Dict[str, Any]:
        """Returns a dictionary describing the optimizer settings
        """
        return {x if x != "name" else "optimizer_name": getattr(self, x) for x in self._setting_names}

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, self.__class__):
            return all(getattr(self, attr) == getattr(other, attr) for attr in self._setting_names)
        return False


def create_seed_generator(seed: Optional[int]) -> Iterator[Optional[int]]:
    """Create a stream of seeds, independent from the standard random stream.
    This is designed to be used in experiment plans generators, fore reproducibility.

    Parameter
    ---------
    seed: int or None
        the initial seed

    Yields
    ------
    int or None
        potential new seeds, or None if the initial seed was None
    """
    generator = None if seed is None else np.random.RandomState(seed=seed)
    while True:
        yield None if generator is None else generator.randint(2**32, dtype=np.uint32)


class Experiment:
    """Specifies an experiment which can be run in benchmarks.

    Parameters
    ----------
    function: BaseFunction
        the function to run the experiment on. It must inherit from BaseFunction to implement
        descriptors for the function.

    Note
    ----
    - "run" method catches error but forwards stderr so that errors are not completely hidden
    - "run" method outputs the description of the experiment, which is a set of figures/names from the functions
    settings (dimension, etc...), the optimization settings (budget, etc...) and the results (loss, etc...)
    """

    # pylint: disable=too-many-arguments
    def __init__(self, function: BaseFunction, optimizer_name: str, budget: int, num_workers: int = 1,
                 batch_mode: bool = True, seed: Optional[int] = None) -> None:
        assert isinstance(function, BaseFunction), "All experiment functions should derive from BaseFunction"
        self.function = function
        self.seed = seed  # depending on the inner workings of the function, the experiment may not be repeatable
        self.optimsettings = OptimizerSettings(name=optimizer_name, num_workers=num_workers, budget=budget, batch_mode=batch_mode)
        self.result = {"loss": np.nan, "elapsed_budget": np.nan, "elapsed_time": np.nan, "error": ""}
        self.recommendation: Optional[base.ArrayLike] = None

    def __repr__(self) -> str:
        return f"Experiment: {self.optimsettings} (dim={self.function.dimension}) on {self.function}"

    @property
    def is_incoherent(self) -> bool:
        """Flags settings which are known to be impossible to process.
        Currently, this means we flag:
        - no_parallelization optimizers for num_workers > 1
        """
        return self.optimsettings.is_incoherent

    def run(self) -> Dict[str, Any]:
        """Run an experiment with the provided settings

        Returns
        -------
        dict
            A dict containing all the information about the experiments (optimizer/function settings + results)

        Note
        ----
        This function catches error (but forwards stderr). It fills up the "error" ("" if no error, else the error name),
        "loss", "elapsed_time" and "elapsed_budget" of the experiment.
        """
        try:
            self._run_with_error()
        except Exception as e:  # pylint: disable=broad-except
            # print the case and the traceback
            self.result["error"] = e.__class__.__name__
            print(f"Error when applying {self}:", file=sys.stderr)
            traceback.print_exc()
            print("\n", file=sys.stderr)
        return self.get_description()

    def _log_results(self, t0: float, num_calls: int) -> None:
        """Internal method for logging results before handling the error
        """
        num_eval = 100  # evaluations of the cost function on the recommendation
        self.result["elapsed_time"] = time.time() - t0
        # make a final evaluation with oracle (no noise, but function may still be stochastic)
        t_recommendation = self.function.transform(self.recommendation)
        self.result["loss"] = sum(self.function.oracle_call(t_recommendation) for _ in range(num_eval)) / num_eval
        self.result["elapsed_budget"] = num_calls
        if num_calls > self.optimsettings.budget:
            raise RuntimeError(f"Too much elapsed budget {num_calls} for {self.optimsettings.name} on {self.function}")

    def _run_with_error(self, callbacks: Optional[Dict[str, base._OptimCallBack]] = None) -> None:
        """Run an experiment with the provided artificial function and optimizer

        Parameter
        ---------
        callbacks: dict
            a dictionary of callbacks to register on the optimizer with key "ask" and/or "tell" (see base Optimizer class).
            This is only for easier debugging.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
        # optimizer instantiation can be slow and is done only here to make xp iterators very fast
        optimizer = self.optimsettings.instanciate(dimension=self.function.dimension)
        if callbacks is not None:
            for name, func in callbacks.items():
                optimizer.register_callback(name, func)
        assert optimizer.budget is not None, "A budget must be provided"
        t0 = time.time()
        counter = CallCounter(self.function)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=base.InefficientSettingsWarning)  # benchmark do not need to be efficient
            # use default executor for batch mode (sequential, but mock for steady state
            # ("production" steady state is a not strictly steady state + does not handle mocked delays)
            executor: Optional[execution.MockedSteadyExecutor] = None if self.optimsettings.batch_mode else execution.MockedSteadyExecutor()
            try:
                self.recommendation = optimizer.optimize(counter, batch_mode=self.optimsettings.batch_mode, executor=executor)
            except Exception as e:  # pylint: disable=broad-except
                self.recommendation = optimizer.provide_recommendation()  # get the recommendation anyway
                self._log_results(t0, counter.num_calls)
                raise e
        self._log_results(t0, counter.num_calls)

    def get_description(self) -> Dict[str, Union[str, float, bool]]:
        """Return the description of the experiment, as a dict.
        "run" must be called beforehand in order to have non-nan values for the loss.
        """
        summary = dict(self.result, seed=-1 if self.seed is None else self.seed)
        summary.update(self.function.descriptors)
        summary.update(self.optimsettings.get_description())
        return summary

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Experiment):
            return False
        return self.function == other.function and self.optimsettings == other.optimsettings
