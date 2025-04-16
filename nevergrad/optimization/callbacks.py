# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import time
import warnings
import inspect
import datetime
import logging
from pathlib import Path
import numpy as np
import nevergrad.common.typing as tp
from nevergrad.common import errors
from nevergrad.parametrization import parameter as p
from nevergrad.parametrization import helpers
from . import base

global_logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------

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

# -------------------------------------------------------------------------------------

class OptimizationLogger:
    """Logger to register as callback in an optimizer, for Logging
    best point regularly.

    Parameters
    ----------
    logger:
        given logger that callback will use to log
    log_level:
        log level that logger will write to
    log_interval_tells: int
        max number of evaluation before performing another log
    log_interval_seconds:
        max number of seconds before performing another log
    """

    def __init__(
        self,
        *,
        logger: logging.Logger = global_logger,
        log_level: int = logging.INFO,
        log_interval_tells: int = 1,
        log_interval_seconds: float = 60.0,
    ) -> None:
        assert log_interval_tells > 0
        assert log_interval_seconds > 0
        self._logger = logger
        self._log_level = log_level
        self._log_interval_tells = int(log_interval_tells)
        self._log_interval_seconds = log_interval_seconds
        self._next_tell = self._log_interval_tells
        self._next_time = time.time() + log_interval_seconds

    def __call__(self, optimizer: base.Optimizer, *args: tp.Any, **kwargs: tp.Any) -> None:
        if time.time() >= self._next_time or self._next_tell >= optimizer.num_tell:
            self._next_time = time.time() + self._log_interval_seconds
            self._next_tell = optimizer.num_tell + self._log_interval_tells
            if optimizer.num_objectives == 1:
                x = optimizer.provide_recommendation()
                self._logger.log(self._log_level, "After %s, recommendation is %s", optimizer.num_tell, x)
            else:
                losses = optimizer._hypervolume_pareto.get_min_losses()  # type: ignore
                self._logger.log(
                    self._log_level,
                    "After %s, the respective minimum loss for each objective in the pareto front is %s",
                    optimizer.num_tell,
                    losses,
                )

# -------------------------------------------------------------------------------------

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
                data.update({"#optimizer#" + x: str(y) for x, y in configopt.config().items()})
        if isinstance(candidate._meta.get("sigma"), float):
            data["#meta-sigma"] = candidate._meta["sigma"]  # for TBPSA-like algorithms
        if candidate.generation > 1:
            data["#parents_uids"] = candidate.parents_uids
        for name, param in helpers.flatten(candidate, with_containers=False, order=1):
            val = param.value
            if isinstance(val, (np.float64, np.int_, np.bool_)):
                val = val.item()
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

# -------------------------------------------------------------------------------------

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

# -------------------------------------------------------------------------------------

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

# -------------------------------------------------------------------------------------

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

    @classmethod
    def timer(cls, max_duration: float) -> "EarlyStopping":
        """Early stop when max_duration seconds has been reached (from the first ask)"""
        return cls(_DurationCriterion(max_duration))

    @classmethod
    def no_improvement_stopper(cls, tolerance_window: int) -> "EarlyStopping":
        """Early stop when loss didn't reduce during tolerance_window asks"""
        return cls(_LossImprovementToleranceCriterion(tolerance_window))

class _DurationCriterion:
    def __init__(self, max_duration: float) -> None:
        self._start = float("inf")
        self._max_duration = max_duration

    def __call__(self, optimizer: base.Optimizer) -> bool:
        if np.isinf(self._start):
            self._start = time.time()
        return time.time() > self._start + self._max_duration

class _LossImprovementToleranceCriterion:
    def __init__(self, tolerance_window: int) -> None:
        self._tolerance_window: int = tolerance_window
        self._best_value: tp.Optional[np.ndarray] = None
        self._tolerance_count: int = 0

    def __call__(self, optimizer: base.Optimizer) -> bool:
        best_param = optimizer.provide_recommendation()
        if best_param is None or (best_param.loss is None and best_param._losses is None):
            return False
        best_last_losses = best_param.losses
        if self._best_value is None:
            self._best_value = best_last_losses
            return False
        if self._best_value <= best_last_losses:
            self._tolerance_count += 1
        else:
            self._tolerance_count = 0
            self._best_value = best_last_losses
        return self._tolerance_count > self._tolerance_window

# -------------------------------------------------------------------------------------

import os
import subprocess
import time
import warnings

class SlurmStopping:
    def __init__(self, threshold_seconds: int = 300):
        self.threshold = threshold_seconds
        self.job_start_time, self.job_duration = self._get_slurm_times()
        self.job_end_time = self.job_start_time + self.job_duration

    def __call__(self, *args, **kwargs):
        if args or kwargs:
            raise errors.NevergradRuntimeError("SlurmStopping must be registered on ask method")
        time_left = self.job_end_time - time.time()
        if time_left <= self.threshold:
            raise errors.NevergradEarlyStopping(f"SLURM timeout in {self.threshold} seconds, stopping optimization.")

    def _get_slurm_times(self):
        job_id = os.environ.get("SLURM_JOB_ID")
        if not job_id:
            warnings.warn("SlurmStopping is a no-op: not running inside a SLURM job.", RuntimeWarning)

        try:
            out = subprocess.check_output(["scontrol", "show", "job", job_id], encoding="utf-8")
        except Exception as e:
            raise errors.NevergradRuntimeError(f"scontrol failed: {e}")

        start_timestamp = None
        time_limit_sec = None
        for line in out.splitlines():
            if "StartTime=" in line:
                # e.g. StartTime=2025-04-15T02:30:00
                for field in line.strip().split():
                    if field.startswith("StartTime="):
                        start_str = field.split("=")[1]
                        if start_str == "Unknown":
                            start_timestamp = time.time()
                        else:
                            start_timestamp = time.mktime(time.strptime(start_str, "%Y-%m-%dT%H:%M:%S"))
            if "TimeLimit=" in line:
                for field in line.strip().split():
                    if field.startswith("TimeLimit="):
                        limit_str = field.split("=")[1]  # e.g. "01:00:00"
                        h, m, s = map(int, limit_str.split(":"))
                        time_limit_sec = h * 3600 + m * 60 + s

        if start_timestamp is None or time_limit_sec is None:
            raise errors.NevergradRuntimeError("Failed to extract StartTime or TimeLimit from scontrol")

        return start_timestamp, time_limit_sec

# -------------------------------------------------------------------------------------

import time
import logging
import numpy as np
from typing import Callable, List
from concurrent.futures import ThreadPoolExecutor, Future
from nevergrad.optimization.base import Optimizer
from nevergrad.parametrization.parameter import Parameter
from nevergrad.common.typing import FloatLoss

logger = logging.getLogger(__name__)

class TimedCallback:
    def __init__(self, interval_sec: float = 60.0):
        self.interval_sec = interval_sec
        self._last_time = time.time()

    def should_run(self) -> bool:
        now = time.time()
        if now - self._last_time >= self.interval_sec:
            self._last_time = now
            return True
        return False

    def get_callback(self) -> Callable[[Optimizer, Parameter, FloatLoss], None]:
        raise NotImplementedError


class HSICLoggerCallback(TimedCallback):
    def __init__(self, parameter_names: tp.Optional[tp.List[str]] = None, kernel_dim=100, sigma=1.0, interval_sec=60.0):
        super().__init__(interval_sec)
        self.kernel_dim = kernel_dim
        self.sigma = sigma
        self._zx_mean = None
        self._zy_mean = None
        self._c_xy = None
        self._n = 0
        self._buffer: List[tuple[np.ndarray, float]] = []
        self._rff_ready = False
        self.rff_weights_x = None
        self.rff_bias_x = None
        self.rff_weights_y = None
        self.rff_bias_y = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._flush_future: Future = None

    def __call__(self, optimizer, candidate, loss):
        self.buffer_point(np.asarray(candidate.value), loss)
        if self.should_run():
            if self._flush_future and not self._flush_future.done():
                logger.warning("Skipped HSIC flush: previous flush still in progress.")
            else:
                self._flush_future = self._executor.submit(self._flush_impl)

    def _init_rff(self, dim_x: int, dim_y: int):
        self.rff_weights_x = np.random.normal(0, 1.0 / self.sigma, (dim_x, self.kernel_dim))
        self.rff_bias_x = np.random.uniform(0, 2 * np.pi, self.kernel_dim)
        self.rff_weights_y = np.random.normal(0, 1.0 / self.sigma, (dim_y, self.kernel_dim))
        self.rff_bias_y = np.random.uniform(0, 2 * np.pi, self.kernel_dim)
        self._rff_ready = True

    def _rff(self, x: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
        return np.sqrt(2.0 / self.kernel_dim) * np.cos(x @ weights + bias)

    def buffer_point(self, x: np.ndarray, y: float):
        self._buffer.append((x.copy(), float(y)))

    def _flush_impl(self):
        if not self._buffer:
            return

        x_all = np.array([x for x, _ in self._buffer])
        y_all = np.array([[y] for _, y in self._buffer])
        self._buffer.clear()

        if not self._rff_ready:
            self._init_rff(x_all.shape[1], y_all.shape[1])

        zx_all = self._rff(x_all, self.rff_weights_x, self.rff_bias_x)
        zy_all = self._rff(y_all, self.rff_weights_y, self.rff_bias_y)

        for zx, zy in zip(zx_all, zy_all):
            zx = zx.reshape(1, -1)
            zy = zy.reshape(1, -1)
            if self._n == 0:
                self._zx_mean = zx.copy()
                self._zy_mean = zy.copy()
                self._c_xy = np.zeros((self.kernel_dim, self.kernel_dim))
            else:
                alpha = 1.0 / (self._n + 1)
                self._zx_mean = (1 - alpha) * self._zx_mean + alpha * zx
                self._zy_mean = (1 - alpha) * self._zy_mean + alpha * zy
                dx = zx - self._zx_mean
                dy = zy - self._zy_mean
                self._c_xy = (1 - alpha) * self._c_xy + alpha * (dx.T @ dy)
            self._n += 1

        print(self.summary())

    def hsic_score(self) -> float:
        return float(np.sum(self._c_xy ** 2)) if self._c_xy is not None else 0.0

    def summary(self) -> str:
        return f"Incremental RFF-HSIC: N={self._n}, HSIC={self.hsic_score():.6f}"

# -------------------------------------------------------------------------------------
