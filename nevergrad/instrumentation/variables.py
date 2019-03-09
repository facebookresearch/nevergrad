# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, TypeVar, Union
import numpy as np
from . import discretization
from ..common.typetools import ArrayLike
from . import utils


X = TypeVar("X")


class SoftmaxCategorical(utils.Variable[X]):
    """Discrete set of n values transformed to a n-dim continuous variable.
    Each of the dimension encodes a weight for a value, and the softmax of weights
    provide probabilities for each possible value. A random value is sampled from
    this distribution.

    Parameter
    ---------
    possibilities: list
        a list of possible values for the variable

    Note
    ----
    Since the chosen value is drawn randomly, the use of this variable makes deterministic
    functions become stochastic, hence "adding noise"
    """

    def __init__(self, possibilities: List[X], deterministic: bool = False) -> None:
        self.deterministic = deterministic
        self.possibilities = list(possibilities)

    @property
    def dimension(self) -> int:
        return len(self.possibilities)

    def process(self, data: ArrayLike, deterministic: bool = False) -> X:
        assert len(data) == len(self.possibilities)
        deterministic = deterministic | self.deterministic
        index = int(discretization.softmax_discretization(data, len(self.possibilities), deterministic=deterministic)[0])
        return self.possibilities[index]

    def process_arg(self, arg: X) -> ArrayLike:
        assert arg in self.possibilities, f'{arg} not in allowed values: {self.possibilities}'
        return discretization.inverse_softmax_discretization(self.possibilities.index(arg), len(self.possibilities))

    def get_summary(self, data: ArrayLike) -> str:
        output = self.process(data, deterministic=True)
        probas = discretization.softmax_probas(np.array(data, copy=False))
        proba_str = ", ".join([f'"{s}": {round(100 * p)}%' for s, p in zip(self.possibilities, probas)])
        return f"Value {output}, from data: {data} yielding probas: {proba_str}"

    def _short_repr(self) -> str:
        return "SC({}|{})".format(",".join([str(x) for x in self.possibilities]), int(self.deterministic))


class OrderedDiscrete(SoftmaxCategorical[X]):
    """Discrete list of n values transformed to a 1-dim discontinuous variable.
    A gaussian input yields a uniform distribution on the list of variables.

    Parameter
    ---------
    possibilities: list
        a list of possible values for the variable

    Note
    ----
    The variables are assumed to be ordered.
    """

    @property
    def dimension(self) -> int:
        return 1

    def process(self, data: ArrayLike, deterministic: bool = False) -> X:  # pylint: disable=arguments-differ, unused-argument
        assert len(data) == 1
        index = discretization.threshold_discretization(data, arity=len(self.possibilities))[0]
        return self.possibilities[index]

    def process_arg(self, arg: X) -> ArrayLike:
        assert arg in self.possibilities, f'{arg} not in allowed values: {self.possibilities}'
        index = self.possibilities.index(arg)
        return discretization.inverse_threshold_discretization([index], len(self.possibilities))

    def get_summary(self, data: ArrayLike) -> str:
        output = self.process(data, deterministic=True)
        return f"Value {output}, from data: {data[0]}"

    def _short_repr(self) -> str:
        return "OD({})".format(",".join([str(x) for x in self.possibilities]))


Y = Union[float, np.ndarray]


class Gaussian(utils.Variable[Y]):
    """Gaussian variable with a mean and a standard deviation, and
    possibly a shape (when using directly in Python)
    The output will simply be mean + std * data
    """

    def __init__(self, mean: float, std: float, shape: Optional[List[int]] = None) -> None:
        self.mean = mean
        self.std = std
        self.shape = shape

    @property
    def dimension(self) -> int:
        return 1 if self.shape is None else int(np.prod(self.shape))

    def process(self, data: ArrayLike, deterministic: bool = True) -> Y:
        assert len(data) == self.dimension
        x = data[0] if self.shape is None else np.reshape(data, self.shape)
        return self.std * x + self.mean

    def process_arg(self, arg: Y) -> ArrayLike:
        return [(arg - self.mean) / self.std]

    def get_summary(self, data: ArrayLike) -> str:
        output = self.process(data)
        return f"Value {output}, from data: {data}"

    def _short_repr(self) -> str:
        return f"G({self.mean},{self.std})"


class _Constant(utils.Variable[X]):
    """Fake variable so that constant variables can fit into the
    pipeline.
    """

    def __init__(self, value: X) -> None:
        self.value = value

    @classmethod
    def convert_non_instrument(cls, x: Union[X, utils.Variable[X]]) -> utils.Variable[X]:
        return x if isinstance(x, utils.Variable) else cls(x)

    @property
    def dimension(self) -> int:
        return 0

    def process(self, data: ArrayLike, deterministic: bool = False) -> X:  # pylint: disable=unused-argument
        return self.value

    def process_arg(self, arg: X) -> ArrayLike:
        assert arg == self.value, f'{arg} != {self.value}'
        return []

    def get_summary(self, data: ArrayLike) -> str:
        raise RuntimeError("Constant summary should not be called")

    def _short_repr(self) -> str:
        return f"{self.value}"
