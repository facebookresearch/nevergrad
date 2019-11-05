# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import time
import warnings
import datetime
from typing import Optional, Any, Union, List, Dict
from pathlib import Path
import numpy as np
from . import base


class OptimizationPrinter:
    """Printer to register as callback in an optimizer, for printing
    best point regularly.

    Parameters
    ----------
    num_tell_period: int
        max number of evaluation before performing another print
    time_period_s: float
        max number of seconds before performing another print
    """

    def __init__(self, num_tell_period: int = 0, time_period_s: float = 60) -> None:
        assert num_tell_period >= 0
        self._num_tell_period = int(num_tell_period)
        self._last_time: Optional[float] = None
        self._time_period_s = time_period_s

    def __call__(self, optimizer: base.Optimizer, *args: Any, **kwargs: Any) -> None:
        if self._last_time is None:
            self._last_time = time.time()
        if ((time.time() - self._last_time) > self._time_period_s or
                (self._num_tell_period and not optimizer.num_tell % self._num_tell_period)):
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
    arrays are converted to lists
    """

    def __init__(self, filepath: Union[str, Path]) -> None:
        self._session = datetime.datetime.now().strftime("%y-%m-%d %H:%M")
        self._filepath = Path(filepath)

    def __call__(self, optimizer: base.Optimizer, candidate: base.Candidate, value: float) -> None:
        data = {"#instrumentation": optimizer.instrumentation.name,
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

    def load(self, max_list_elements: int = 24) -> List[Dict[str, Any]]:
        """

        Parameters
        ----------
        max_list_elements: int
            Maximum number of elements displayed from the array, each element is given a
            unique id of type list_name#i1_i2_...
        """
        data: List[Dict[str, Any]] = []
        with self._filepath.open("r") as f:
            for line in f.readlines():
                data.append(json.loads(line))
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
