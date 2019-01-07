# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time
import random
import warnings
import traceback
from typing import Dict, Union, Callable, Any, Optional, Iterator
import numpy as np
from ..common import decorators
from ..functions import BaseFunction
from ..optimization import base
from ..optimization.optimizerlib import registry as optimizer_registry


registry = decorators.Registry()


class CallCounter:
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
        yield None if generator is None else generator.randint(np.iinfo(np.uint32).max)


class Experiment:
    """Specifies an experiment which can be run in benchmarks.

    Parameters
    ----------
    function: BaseFunction
        the function to run the experiment on. It must inherit from BaseFunction so that to implement
        descriptors for the function.

    Note
    ----
    - "run" method catches error but forwards stderr so that errors are not completely hidden
    - "run" method outputs the description of the experiment, which is a set of figures/names from the functions
    settings (dimension, etc...), the optimization settings (budget, etc...) and the results (loss, etc...)
    """

    # pylint: disable=too-many-arguments
    def __init__(self, function: BaseFunction, optimizer_name: str, budget: int, num_workers: int = 1, seed: Optional[int] = None) -> None:
        assert isinstance(function, BaseFunction), "All experiment functions should derive from BaseFunction"
        self.function = function
        self.seed = seed  # depending on the inner workings of the function, the experiment may not be repeatable
        assert optimizer_name in optimizer_registry, f"{optimizer_name} is not registered"
        self._optimizer_parameters = {"optimizer_name": optimizer_name, "num_workers": num_workers, "budget": budget}
        self.result = {"loss": np.nan, "elapsed_budget": np.nan, "elapsed_time": np.nan, "error": ""}

    def __repr__(self) -> str:
        budget, num_workers, optimizer_name = [self._optimizer_parameters[x] for x in ["budget", "num_workers", "optimizer_name"]]
        dim = self.function.dimension
        return f"Experiment: {optimizer_name}(dimension={dim}, budget={budget}, num_workers={num_workers}) on {self.function}"

    @property
    def is_incoherent(self) -> bool:
        """Flags settings which are known to be impossible to process.
        Currently, this means we flag:
        - no_parallelization optimizers for num_workers > 1
        """
        # flag no_parallelization when num_workers greater than 1
        optimizer = optimizer_registry[self._optimizer_parameters["optimizer_name"]]
        return optimizer.no_parallelization and bool(self._optimizer_parameters["num_workers"] > 1)  # type: ignore

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

    def _run_with_error(self) -> None:
        """Run an experiment with the provided artificial function and optmizer
        """
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
        budget, num_workers, optimizer_name = [self._optimizer_parameters[x] for x in ["budget", "num_workers", "optimizer_name"]]
        # optimizer instantiation can be slow and is done only here to make xp iterators very fast
        optimizer = optimizer_registry[optimizer_name](dimension=self.function.dimension, budget=budget, num_workers=num_workers)
        assert optimizer.budget is not None, "A budget must be provided"
        assert optimizer.dimension == self.function.dimension
        t0 = time.time()
        counter = CallCounter(self.function)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=base.InefficientSettingsWarning)  # benchmark do not need to be efficient
            recommendation = optimizer.optimize(counter, batch_mode=True)
        self.result["elapsed_time"] = time.time() - t0
        # make a final evaluation with oracle (no noise, but function may still be stochastic)
        num_eval = 100
        self.result["loss"] = sum(self.function.oracle_call(recommendation) for _ in range(num_eval)) / num_eval
        self.result["elapsed_budget"] = counter.num_calls
        if self.result["elapsed_budget"] > optimizer.budget:
            raise RuntimeError(f"Too much elapsed budget {self.result['elapsed_budget']} for {optimizer} on {self.function}")

    def get_description(self) -> Dict[str, Union[str, float, bool]]:
        """Return the description of the experiment, as a dict.
        "run" must be called beforehand in order to have non-nan values for the loss.
        """
        summary = dict(self.result, seed=-1 if self.seed is None else self.seed)
        summary.update(self.function.descriptors)
        summary.update(self._optimizer_parameters)
        return summary

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Experiment):
            return False
        return self.function == other.function and self._optimizer_parameters == other._optimizer_parameters
