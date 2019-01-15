# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re
import itertools
from typing import List, Any, Match, Optional, Tuple, Dict
import numpy as np
from ..optimization import discretization
from ..common.typetools import ArrayLike
from . import utils


class _Variable(utils.Instrument):
    """Base variable class.
    Each class requires to provide a dimension and ways to process the data.
    They can be used directly for function instrumentation in Python, but they
    must also provide token management for hardcoded code instrumentation.
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

    def process(self, data: ArrayLike, deterministic: bool = False) -> Any:
        assert len(data) == len(self.possibilities)
        index = int(discretization.softmax_discretization(data, len(self.possibilities), deterministic=deterministic)[0])
        return self.possibilities[index]

    def process_arg(self, arg: Any) -> ArrayLike:
        assert arg in self.possibilities, f'{arg} not in allowed values: {self.possibilities}'
        return discretization.inverse_softmax_discretization(self.possibilities.index(arg), len(self.possibilities))

    def get_summary(self, data: List[float]) -> str:
        output = self.process(data, deterministic=True)
        probas = discretization.softmax_probas(data)
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
        index = discretization.threshold_discretization(data, arity=len(self.possibilities))[0]
        return self.possibilities[index]

    def process_arg(self, arg: Any) -> ArrayLike:
        assert arg in self.possibilities, f'{arg} not in allowed values: {self.possibilities}'
        index = self.possibilities.index(arg)
        return discretization.inverse_threshold_discretization([index], len(self.possibilities))

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

    def process(self, data: List[float], deterministic: bool = True) -> Any:
        assert len(data) == self.dimension
        x = data[0] if self.shape is None else np.reshape(data, self.shape)
        return self.std * x + self.mean

    def process_arg(self, arg: Any) -> List[float]:
        return [(arg - self.mean) / self.std]

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
    def convert_non_instrument(cls, x: Any) -> utils.Instrument:
        return x if isinstance(x, utils.Instrument) else cls(x)

    @property
    def dimension(self) -> int:
        return 0

    def process(self, data: List[float], deterministic: bool = False) -> Any:  # pylint: disable=unused-argument
        return self._value

    def process_arg(self, arg: Any) -> ArrayLike:
        assert arg == self._value, f'{arg} != {self._value}'
        return []

    def get_summary(self, data: List[float]) -> str:
        raise RuntimeError("Constant summary should not be called")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._value})"


class Instrumentation:
    """Class handling arguments instrumentation, and providing conversion to and from data.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.names, arguments = self._make_argument_names_and_list(*args, **kwargs)
        self.instruments: List[utils.Instrument] = [_Constant.convert_non_instrument(a) for a in arguments]

    @property
    def dimension(self) -> int:
        return sum(i.dimension for i in self.instruments)

    @staticmethod
    def _make_argument_names_and_list(*args: Any, **kwargs: Any) -> Tuple[Tuple[Optional[str], ...], Tuple[Any, ...]]:
        """Converts *args and **kwargs to a tuple of names (with None for positional),
        and the corresponding tuple of values.

        Eg:
        _make_argument_names_and_list(3, z="blublu", machin="truc")
        >>> (None, "machin", "z"), (3, "truc", "blublu")
        """
        names: Tuple[Optional[str], ...] = tuple([None] * len(args) + sorted(kwargs))  # type: ignore
        arguments: Tuple[Any, ...] = args + tuple(kwargs[x] for x in names if x is not None)
        return names, arguments

    def data_to_arguments(self, data: np.ndarray, deterministic: bool = True) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """Converts data to arguments
        """
        arguments = utils.process_instruments(self.instruments, data, deterministic=deterministic)
        args = tuple(arg for name, arg in zip(self.names, arguments) if name is None)
        kwargs = {name: arg for name, arg in zip(self.names, arguments) if name is not None}
        return args, kwargs

    def arguments_to_data(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Converts arguments to data

        Note
        ----
        - you need to input the arguments in the same way than at initialization
          with regard to positional and named arguments.
        - this process is simplified, and is deterministic. Depending on your instrumentation,
          you will probably not recover the same data.
        """
        names, arguments = self._make_argument_names_and_list(*args, **kwargs)
        assert names == self.names, (f"Passed argument pattern (positional Vs named) was:\n{names}\n"
                                     f"but expected:\n{self.names}")
        data = list(itertools.chain.from_iterable([instrument.process_arg(arg) for instrument, arg in zip(self.instruments, arguments)]))
        return np.array(data)
