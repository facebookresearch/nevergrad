# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Definitions of some convenient types.
If you know better practices, feel free to submit it ;)
"""
from typing import Union, Tuple, List, Any, Callable, TypeVar
from pathlib import Path
from typing_extensions import Protocol
import numpy as np


ArrayLike = Union[Tuple[float, ...], List[float], np.ndarray]
PathLike = Union[str, Path]


# %% Protocol definitions for executor typing

X = TypeVar('X', covariant=True)


class JobLike(Protocol[X]):
    # pylint: disable=pointless-statement

    def done(self) -> bool:
        ...

    def result(self) -> X:
        ...


class ExecutorLike(Protocol):
    # pylint: disable=pointless-statement, unused-argument

    def submit(self, fn: Callable[..., X], *args: Any, **kwargs: Any) -> JobLike[X]:
        ...
