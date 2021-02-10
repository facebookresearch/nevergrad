# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Definitions of some convenient types.
If you know better practices, feel free to submit it ;)
"""
# pylint: disable=unused-import
# structures
from typing import Any as Any
from typing import Generic as Generic
from typing import Type as Type
from typing import TypeVar as TypeVar
from typing import Optional as Optional
from typing import Union as Union

# containers
from typing import Dict as Dict
from typing import Tuple as Tuple
from typing import List as List
from typing import Set as Set
from typing import Deque as Deque
from typing import Sequence as Sequence
from typing import NamedTuple as NamedTuple
from typing import MutableMapping as MutableMapping

# iterables
from typing import Iterator as Iterator
from typing import Iterable as Iterable
from typing import Generator as Generator
from typing import KeysView as KeysView
from typing import ValuesView as ValuesView
from typing import ItemsView as ItemsView

# others
from typing import Callable as Callable
from typing import Hashable as Hashable
from typing import Match as Match
from pathlib import Path as Path
from typing_extensions import Protocol

#
import numpy as _np


ArgsKwargs = Tuple[Tuple[Any, ...], Dict[str, Any]]
ArrayLike = Union[Tuple[float, ...], List[float], _np.ndarray]
PathLike = Union[str, Path]
FloatLoss = float
Loss = Union[float, ArrayLike]
BoundValue = Optional[Union[float, int, _np.int, _np.float, _np.ndarray]]


# %% Protocol definitions for executor typing

X = TypeVar("X", covariant=True)


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
