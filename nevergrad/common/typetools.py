# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Definitions of some convenient types.
If you know better practices, feel free to submit it ;)
"""
from typing import Union, Tuple, Any, Callable
from pathlib import Path
from typing_extensions import Protocol
import numpy as np


ArrayLike = Union[Tuple[float, ...], np.ndarray]
PathLike = Union[str, Path]


# %% Protocol definitions for executor typing

class JobLike(Protocol):
    # pylint: disable=pointless-statement

    def done(self) -> bool:
        ...

    def result(self) -> Any:
        ...


class ExecutorLike(Protocol):
    # pylint: disable=pointless-statement, unused-argument

    def submit(self, function: Callable, *args: Any, **kwargs: Any) -> JobLike:
        ...
