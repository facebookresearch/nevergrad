# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import time
import warnings
import inspect
import datetime
from pathlib import Path
import numpy as np
import nevergrad.common.typing as tp
from nevergrad.common import errors
from nevergrad.parametrization import parameter as p
from nevergrad.parametrization import helpers
from . import base


class OptimizationPrinter:
    """Printer to register as callback in an optimizer, for printing
    best point regularly.

    Parameters
    ----------
    print_interval_tells: int
        max number of evaluation before performing another print
    print_interval_seconds: float
        max number of seconds before performing another print
    """

    def __init__(self, print_interval_tells: int = 1, print_interval_seconds: float = 60.0) -> None:
        assert print_interval_tells > 0
        assert print_interval_seconds > 0
        self._print_interval_tells = int(print_interval_tells)
        self._print_interval_seconds = print_interval_seconds
        self._next_tell = self._print_interval_tells
        self._next_time = time.time() + print_interval_seconds

    def __call__(self, optimizer: base.Optimizer, *args: tp.Any, **kwargs: tp.Any) -> None:
        if time.time() >= self._next_time or self._next_tell >= optimizer.num_tell:
            self._next_time = time.time() + self._print_interval_seconds
            self._next_tell = optimizer.num_tell + self._print_interval_tells
            x = optimizer.provide_recommendation()
            print(f"After {optimizer.num_tell}, recommendation is {x}")  # TODO fetch value


class ParametersLogger:
    """Logs parameter and run information throughout into a file during
    optimization.

    Parameters
    ----------
    filepath: str or pathlib.Path
        the path to dump data to
    append: bool
        whether to append the file (otherwise it replaces it)
    order: int
        order of the internal/model parameters to extract

    Example
    -------

    .. code-block:: python

        logger = ParametersLogger(filepath)
        optimizer.register_callback("tell",  logger)
        optimizer.minimize()
        list_of_dict_of_data = logger.load()

    Note
    ----
    Arrays are converted to lists
    """

    def __init__(self, filepath: tp.Union[str, Path], append: bool = True, order: int = 1) -> None:
        self._session = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
        self._filepath = Path(filepath)
        self._order = order
        if self._filepath.exists() and not append:
            self._filepath.unlink()  # missing_ok argument added in python 3.8
        self._filepath.parent.mkdir(exist_ok=True, parents=True)

    def __call__(self, optimizer: base.Optimizer, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        data = {
            "#parametrization": optimizer.parametrization.name,
            "#optimizer": optimizer.name,
            "#session": self._session,
            "#num-ask": optimizer.num_ask,
            "#num-tell": optimizer.num_tell,
            "#num-tell-not-asked": optimizer.num_tell_not_asked,
            "#uid": candidate.uid,
            "#lineage": candidate.heritage["lineage"],
            "#generation": candidate.generation,
            "#parents_uids": [],
            "#loss": loss,
        }
        if optimizer.num_objectives > 1:  # multiobjective losses
            data.update({f"#losses#{k}": val for k, val in enumerate(candidate.losses)})
            data["#pareto-length"] = len(optimizer.pareto_front())
        if hasattr(optimizer, "_configured_optimizer"):
            configopt = optimizer._configured_optimizer  # type: ignore
            if isinstance(configopt, base.ConfiguredOptimizer):
                data.update({"#optimizer#" + x: y for x, y in configopt.config().items()})
        if isinstance(candidate._meta.get("sigma"), float):
            data["#meta-sigma"] = candidate._meta["sigma"]  # for TBPSA-like algorithms
        if candidate.generation > 1:
            data["#parents_uids"] = candidate.parents_uids
        for name, param in helpers.flatten(candidate, with_containers=False, order=1):
            val = param.value
            if inspect.ismethod(val):
                val = repr(val.__self__)  # show mutation class
            data[name if name else "0"] = val.tolist() if isinstance(val, np.ndarray) else val
            if isinstance(param, p.Data):
                val = param.sigma.value
                data[(name if name else "0") + "#sigma"] = (
                    val.tolist() if isinstance(val, np.ndarray) else val
                )
        try:  # avoid bugging as much as possible
            with self._filepath.open("a") as f:
                f.write(json.dumps(data) + "\n")
        except Exception as e:  # pylint: disable=broad-except
            warnings.warn(f"Failing to json data: {e}")

    def load(self) -> tp.List[tp.Dict[str, tp.Any]]:
        """Loads data from the log file"""
        data: tp.List[tp.Dict[str, tp.Any]] = []
        if self._filepath.exists():
            with self._filepath.open("r") as f:
                for line in f.readlines():
                    data.append(json.loads(line))
        return data

    def load_flattened(self, max_list_elements: int = 24) -> tp.List[tp.Dict[str, tp.Any]]:
        """Loads data from the log file, and splits lists (arrays) into multiple arguments

        Parameters
        ----------
        max_list_elements: int
            Maximum number of elements displayed from the array, each element is given a
            unique id of type list_name#i0_i1_...
        """
        data = self.load()
        flat_data: tp.List[tp.Dict[str, tp.Any]] = []
        for element in data:
            list_keys = {key for key, val in element.items() if isinstance(val, list)}
            flat_data.append({key: val for key, val in element.items() if key not in list_keys})
            for key in list_keys:
                for k, (indices, value) in enumerate(np.ndenumerate(element[key])):
                    if k >= max_list_elements:
                        break
                    flat_data[-1][key + "#" + "_".join(str(i) for i in indices)] = value
        return flat_data

    def to_hiplot_experiment(
        self, max_list_elements: int = 24
    ) -> tp.Any:  # no typing here since Hiplot is not a hard requirement
        """Converts the logs into an hiplot experiment for display.

        Parameters
        ----------
        max_list_elements: int
            maximum number of elements of list/arrays to export (only the first elements are extracted)

        Example
        -------
        .. code-block:: python

            exp = logs.to_hiplot_experiment()
            exp.display(force_full_width=True)

        Note
        ----
        - You can easily change the axes of the XY plot:
          :code:`exp.display_data(hip.Displays.XY).update({'axis_x': '0#0', 'axis_y': '0#1'})`
        - For more context about hiplot, check:

          - blogpost: https://ai.facebook.com/blog/hiplot-high-dimensional-interactive-plots-made-easy/
          - github repo: https://github.com/facebookresearch/hiplot
          - documentation: https://facebookresearch.github.io/hiplot/
        """
        # pylint: disable=import-outside-toplevel
        try:
            import hiplot as hip
        except ImportError as e:
            raise ImportError(
                f"{self.__class__.__name__} requires hiplot which is not installed by default "
                "(pip install hiplot)"
            ) from e
        exp = hip.Experiment()
        for xp in self.load_flattened(max_list_elements=max_list_elements):
            dp = hip.Datapoint(
                from_uid=xp.get("#parents_uids#0"),
                uid=xp["#uid"],
                values={
                    x: y for x, y in xp.items() if not (x.startswith("#") and ("uid" in x or "ask" in x))
                },
            )
            exp.datapoints.append(dp)
        exp.display_data(hip.Displays.XY).update({"axis_x": "#num-tell", "axis_y": "#loss"})
        # for the record, some more options:
        exp.display_data(hip.Displays.XY).update({"lines_thickness": 1.0, "lines_opacity": 1.0})
        return exp


class OptimizerDump:
    """Dumps the optimizer to a pickle file at every call.

    Parameters
    ----------
    filepath: str or Path
        path to the pickle file
    """

    def __init__(self, filepath: tp.Union[str, Path]) -> None:
        self._filepath = filepath

    def __call__(self, opt: base.Optimizer, *args: tp.Any, **kwargs: tp.Any) -> None:
        opt.dump(self._filepath)


class ProgressBar:
    """Progress bar to register as callback in an optimizer"""

    def __init__(self) -> None:
        self._progress_bar: tp.Any = None
        self._current = 0

    def __call__(self, optimizer: base.Optimizer, *args: tp.Any, **kwargs: tp.Any) -> None:
        if self._progress_bar is None:
            # pylint: disable=import-outside-toplevel
            try:
                from tqdm import tqdm  # Inline import to avoid additional dependency
            except ImportError as e:
                raise ImportError(
                    f"{self.__class__.__name__} requires tqdm which is not installed by default "
                    "(pip install tqdm)"
                ) from e
            self._progress_bar = tqdm()
            self._progress_bar.total = optimizer.budget
            self._progress_bar.update(self._current)
        self._progress_bar.update(1)
        self._current += 1

    def __getstate__(self) -> tp.Dict[str, tp.Any]:
        """Used for pickling (tqdm is not picklable)"""
        state = dict(self.__dict__)
        state["_progress_bar"] = None
        return state


class EarlyStopping:
    """Callback for stopping the :code:`minimize` method before the budget is
    fully used.

    Parameters
    ----------
    stopping_criterion: func(optimizer) -> bool
        function that takes the current optimizer as input and returns True
        if the minimization must be stopped

    Note
    ----
    This callback must be register on the "ask" method only.

    Example
    -------
    In the following code, the :code:`minimize` method will be stopped at the 4th "ask"

    >>> early_stopping = ng.callbacks.EarlyStopping(lambda opt: opt.num_ask > 3)
    >>> optimizer.register_callback("ask", early_stopping)
    >>> optimizer.minimize(_func, verbosity=2)

    A couple other options (equivalent in case of non-noisy optimization) for stopping
    if the loss is below 12:

    >>> early_stopping = ng.callbacks.EarlyStopping(lambda opt: opt.recommend().loss < 12)
    >>> early_stopping = ng.callbacks.EarlyStopping(lambda opt: opt.current_bests["minimum"].mean < 12)
    """

    def __init__(self, stopping_criterion: tp.Callable[[base.Optimizer], bool]) -> None:
        self.stopping_criterion = stopping_criterion

    def __call__(self, optimizer: base.Optimizer, *args: tp.Any, **kwargs: tp.Any) -> None:
        if args or kwargs:
            raise errors.NevergradRuntimeError("EarlyStopping must be registered on ask method")
        if self.stopping_criterion(optimizer):
            raise errors.NevergradEarlyStopping("Early stopping criterion is reached")
