# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import List, Any, Match, Optional
import numpy as np
from ..optimization.discretization import (softmax_discretization,
                                           softmax_probas,
                                           threshold_discretization,
                                           inverse_threshold_discretization)
from ..common.typetools import ArrayLike
from . import utils


class _Variable(utils.Instrument):
    """Base variable class.
    Each class requires to provide a dimension and ways to process the data.
    They can be used directly for function instrumentation in Python, but they
    must also provide a token managements for hardcoded code instrumentation.
    """

    pattern: Optional[str] = None
    example: Optional[str] = None

    @classmethod
    def from_regex(cls, regex: Match) -> '_Variable':
        raise NotImplementedError

    @classmethod
    def from_str(cls, string: str) -> '_Variable':
        if cls.pattern is None:
            raise ValueError("pattern must be provided")
        regex = re.search(cls.pattern, string)
        if regex is None:
            raise RuntimeError("Could not find regex")
        return cls.from_regex(regex)

    def __eq__(self, other: Any) -> bool:
        return bool(self.__class__ == other.__class__ and self.__dict__ == other.__dict__)

    def __repr__(self) -> str:
        args = ", ".join(f"{x}={y}" for x, y in sorted(self.__dict__.items()))
        return f"{self.__class__.__name__}({args})"


@utils.vartypes.register
class SoftmaxCategorical(_Variable):
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

    pattern = r'NG_SC' + r'{(?P<possibilities>.*?\|.*?)}'
    example = 'NG_SC{p1|p2|p3...}'

    def __init__(self, possibilities: List[Any]) -> None:
        self.possibilities = list(possibilities)

    @classmethod
    def from_regex(cls, regex: Match) -> _Variable:
        return cls(regex.group("possibilities").split("|"))

    @property
    def dimension(self) -> int:
        return len(self.possibilities)

    def process(self, data: ArrayLike, deterministic: bool = False) -> Any:  # pylint: disable=arguments-differ
        assert len(data) == len(self.possibilities)
        index = int(softmax_discretization(data, len(self.possibilities), deterministic=deterministic)[0])
        return self.possibilities[index]

    def process_arg(self, arg: Any) -> ArrayLike:
        assert arg in self.possibilities, f'{arg} not in allowed values: {self.possibilities}'
        # TODO: Move to nevergrad.optimizatino.discretization.inverse_softmax_discretization

        def inverse_softmax_discretization(index: int, arity: int) -> ArrayLike:
            # p is an arbitrary probability that the provided arg will be sampled with the returned point
            p = (1 / arity) * 1.5
            x = np.zeros(arity)
            x[index] = np.log((p * (arity - 1)) / (1 - p))
            return x

        return inverse_softmax_discretization(self.possibilities.index(arg), len(self.possibilities))

    def get_summary(self, data: List[float]) -> str:
        output = self.process(data, deterministic=True)
        probas = softmax_probas(data)
        proba_str = ", ".join([f'"{s}": {round(100 * p)}%' for s, p in zip(self.possibilities, probas)])
        return f"Value {output}, from data: {data} yielding probas: {proba_str}"


@utils.vartypes.register
class OrderedDiscrete(SoftmaxCategorical):
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

    pattern = r'NG_OD' + r'{(?P<possibilities>.*?\|.*?)}'
    example = 'NG_OD{p1|p2|p3...}'

    @property
    def dimension(self) -> int:
        return 1

    def process(self, data: ArrayLike, deterministic: bool = False) -> Any:  # pylint: disable=arguments-differ, unused-argument
        assert len(data) == 1
        index = threshold_discretization(data, arity=len(self.possibilities))[0]
        return self.possibilities[index]

    def process_arg(self, arg: Any) -> ArrayLike:
        assert arg in self.possibilities, f'{arg} not in allowed values: {self.possibilities}'
        index = self.possibilities.index(arg)
        return inverse_threshold_discretization([index], len(self.possibilities))

    def get_summary(self, data: List[float]) -> str:
        output = self.process(data, deterministic=True)
        return f"Value {output}, from data: {data[0]}"


@utils.vartypes.register
class Gaussian(_Variable):
    """Gaussian variable with a mean and a standard deviation, and
    possibly a shape (when using directly in Python)
    The output will simply be mean + std * data
    """

    pattern = r'NG_G' + r'{(?P<mean>.*?),(?P<std>.*?)}'
    example = 'NG_G{1,2}'

    def __init__(self, mean: float, std: float, shape: Optional[List[int]] = None) -> None:
        self.mean = mean
        self.std = std
        self.shape = shape

    @classmethod
    def from_regex(cls, regex: Match) -> _Variable:
        return cls(float(regex.group("mean")), float(regex.group("std")))

    @property
    def dimension(self) -> int:
        return 1 if self.shape is None else int(np.prod(self.shape))

    def process(self, data: List[float]) -> Any:
        assert len(data) == self.dimension
        x = data[0] if self.shape is None else np.reshape(data, self.shape)
        return self.std * x + self.mean

    def get_summary(self, data: List[float]) -> str:
        output = self.process(data)
        return f"Value {output}, from data: {data}"


class _Constant(utils.Instrument):
    """Fake variable so that constant variables can fit into the
    pipeline.
    """

    def __init__(self, value: Any) -> None:
        self._value = value

    @classmethod
    def convert_non_token(cls, x: Any) -> utils.Instrument:
        return x if isinstance(x, _Variable) else cls(x)

    @property
    def dimension(self) -> int:
        return 0

    def process(self, data: List[float]) -> Any:  # pylint: disable=unused-argument
        return self._value

    def process_arg(self, arg: Any) -> ArrayLike:
        assert arg == self._value, f'{arg} != {self._value}'
        return []

    def get_summary(self, data: List[float]) -> str:
        raise RuntimeError("Constant summary should not be called")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._value})"
