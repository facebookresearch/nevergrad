# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import time
import warnings
import datetime
from typing import Any, Union, List, Dict
from pathlib import Path
import numpy as np
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

    def __call__(self, optimizer: base.Optimizer, *args: Any, **kwargs: Any) -> None:
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

    def __init__(self, filepath: Union[str, Path], delete_existing_file: bool = False) -> None:
        self._session = datetime.datetime.now().strftime("%y-%m-%d %H:%M")
        self._filepath = Path(filepath)
        if self._filepath.exists() and delete_existing_file:
            self._filepath.unlink()  # missing_ok argument added in python 3.8

    def __call__(self, optimizer: base.Optimizer, candidate: base.Candidate, value: float) -> None:
        data = {"#instrumentation": optimizer.instrumentation.name,
                "#name": optimizer.name,
                "#session": self._session,
                "#num-ask": optimizer.num_ask,
                "#num-tell": optimizer.num_tell,
                "#num-tell-not-asked": optimizer.num_tell_not_asked,
                "#loss": value}
        params = dict(candidate.kwargs)
        params.update({f"#arg{k}": arg for k, arg in enumerate(candidate.args)})
        data.update({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in params.items()})
        try:  # avoid bugging as much as possible
            with self._filepath.open("a") as f:
                f.write(json.dumps(data) + "\n")
        except Exception:  # pylint: disable=broad-except
            warnings.warn("Failing to json data")

    def load(self) -> List[Dict[str, Any]]:
        """Loads data from the log file
        """
        data: List[Dict[str, Any]] = []
        if self._filepath.exists():
            with self._filepath.open("r") as f:
                for line in f.readlines():
                    data.append(json.loads(line))
        return data

    def load_flattened(self, max_list_elements: int = 24) -> List[Dict[str, Any]]:
        """Loads data from the log file, and splits lists (arrays) into multiple arguments

        Parameters
        ----------
        max_list_elements: int
            Maximum number of elements displayed from the array, each element is given a
            unique id of type list_name#i1_i2_...
        """
        data = self.load()
        flat_data: List[Dict[str, Any]] = []
        for element in data:
            list_keys = {key for key, val in element.items() if isinstance(val, list)}
            flat_data.append({key: val for key, val in element.items() if key not in list_keys})
            for key in list_keys:
                for k, (indices, value) in enumerate(np.ndenumerate(element[key])):
                    if k >= max_list_elements:
                        break
                    flat_data[-1][key + "#" + "_".join(str(i) for i in indices)] = value
        return flat_data
