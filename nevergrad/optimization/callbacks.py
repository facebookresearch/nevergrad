# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import time
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
    num_eval: int
        max number of evaluation before performing another print
    num_sec: float
        max number of seconds before performing another print
    """

    def __init__(self, num_eval: int = 0, num_sec: float = 60) -> None:
        self._num_eval = max(0, int(num_eval))
        self._last_time: Optional[float] = None
        self._num_sec = num_sec

    def __call__(self, optimizer: base.Optimizer, *args: Any, **kwargs: Any) -> None:
        if self._last_time is None:
            self._last_time = time.time()
        if (time.time() - self._last_time) > self._num_sec or (self._num_eval and not optimizer.num_tell % self._num_eval):
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
        params.update({f"arg{k}": arg for k, arg in enumerate(candidate.args)})
        data.update({x: y for x, y in params.items() if not isinstance(y, np.ndarray)})
        data.update({x + "_0": y.ravel()[0] for x, y in params.items() if isinstance(y, np.ndarray)})
        with self._filepath.open("a") as f:
            f.write(json.dumps(data) + "\n")

    def load(self) -> List[Dict[str, Any]]:
        data: List[Dict[str, Any]] = []
        with self._filepath.open("r") as f:
            for line in f.readlines():
                data.append(json.loads(line))
        return data
