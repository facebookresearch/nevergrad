# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import time
import warnings
import datetime
import typing as tp
from pathlib import Path
import numpy as np
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

    Usage
    -----
    logger = ParametersLogger(filepath)
    optimizer.register_callback("tell",  logger)
    optimizer.minimize()
    list_of_dict_of_data = logger.load()

    Note
    ----
    - arrays are converted to lists
    - this class will eventually contain display methods
    """

    def __init__(self, filepath: tp.Union[str, Path], append: bool = True, order: int = 1) -> None:
        self._session = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
        self._filepath = Path(filepath)
        self._order = order
        if self._filepath.exists() and not append:
            self._filepath.unlink()  # missing_ok argument added in python 3.8

    def __call__(self, optimizer: base.Optimizer, candidate: p.Parameter, value: float) -> None:
        data = {"#parametrization": optimizer.parametrization.name,
                "#optimizer": optimizer.name,
                "#session": self._session,
                "#num-ask": optimizer.num_ask,
                "#num-tell": optimizer.num_tell,
                "#num-tell-not-asked": optimizer.num_tell_not_asked,
                "#uid": candidate.uid,
                "#generation": candidate.generation,
                "#parents_uids": [],
                "#loss": value}
        if hasattr(optimizer, "_parameters"):
            configopt = optimizer._parameters  # type: ignore
            if isinstance(configopt, base.ParametrizedFamily):
                data.update({"#optimizer#" + x: y for x, y in configopt.config().items()})
        if candidate.generation > 1:
            data["#parents_uids"] = candidate.parents_uids
        for name, param in helpers.flatten_parameter(candidate, with_containers=False, order=1).items():
            val = param.value
            data[name if name else "0"] = val.tolist() if isinstance(val, np.ndarray) else val
            if isinstance(param, p.Array):
                val = param.sigma.value
                data[(name if name else "0") + "#sigma"] = val.tolist() if isinstance(val, np.ndarray) else val
        try:  # avoid bugging as much as possible
            with self._filepath.open("a") as f:
                f.write(json.dumps(data) + "\n")
        except Exception:  # pylint: disable=broad-except
            warnings.warn("Failing to json data")

    def load(self) -> tp.List[tp.Dict[str, tp.Any]]:
        """Loads data from the log file
        """
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

    def to_hiplot_experiment(self, max_list_elements: int = 24) -> tp.Any:  # no typing here since Hiplot is not a hard requirement
        """Converts the logs into an hiplot experiment for display.


        Example
        -------
        exp = logs.to_hiplot_experiment()
        exp.display(force_full_width=True)

        Note
        ----
        - You can easily change the axes of the XY plot:
          exp.display_data(hip.Displays.XY).update({'axis_x': '0#0', 'axis_y': '0#1'})
        - For more context about hiplot, check:
          - blogpost: https://ai.facebook.com/blog/hiplot-high-dimensional-interactive-plots-made-easy/
          - github repo: https://github.com/facebookresearch/hiplot
          - documentation: https://facebookresearch.github.io/hiplot/
        """
        import hiplot as hip
        exp = hip.Experiment()
        for xp in self.load_flattened(max_list_elements=max_list_elements):
            dp = hip.Datapoint(
                from_uid=xp.get("#parents_uids#0"),
                uid=xp["#uid"],
                values={x: y for x, y in xp.items() if not (x.startswith("#") and ("uid" in x or "ask" in x))}
            )
            exp.datapoints.append(dp)
        exp.display_data(hip.Displays.XY).update({'axis_x': '#num-tell', 'axis_y': '#loss'})
        # for the record, some more options:
        exp.display_data(hip.Displays.XY).update({'lines_thickness': 1.0, 'lines_opacity': 1.0})
        return exp
