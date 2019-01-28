# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from typing import List, Any, Match, Optional, Tuple, Dict, Callable
import numpy as np
from . import discretization
from ..functions import base
from ..common.typetools import ArrayLike
from . import utils


_Variable = utils.Instrument


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

    @property
    def args(self) -> Tuple[utils.Instrument, ...]:
        """List of instruments passed as positional arguments
        """
        return tuple(arg for name, arg in zip(self.names, self.instruments) if name is None)

    @property
    def kwargs(self) -> Dict[str, utils.Instrument]:
        """Dictionary of instruments passed as named arguments
        """
        return {name: arg for name, arg in zip(self.names, self.instruments) if name is not None}

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


class InstrumentedFunction(base.BaseFunction):
    """Converts a multi-argument function into a mono-argument multidimensional continuous function
    which can be optimized.

    Parameters
    ----------
    function: callable
        the callable to convert
    *args, **kwargs: Any
        Any argument. Arguments of type variables.SoftmaxCategorical or variables.Gaussian will be instrumented
        and others will be kept constant.

    Note
    ----
    - Tokens can be:
      - DiscreteToken(list_of_n_possible_values): converted into a n-dim array, corresponding to proba for each value
      - GaussianToken(mean, std, shape=None): a Gaussian variable (shape=None) or array.
    - This function can then be directly used in benchmarks *if it returns a float*.

    """

    def __init__(self, function: Callable, *args: Any, **kwargs: Any) -> None:
        assert callable(function)
        self.instrumentation = Instrumentation(*args, **kwargs)
        super().__init__(dimension=self.instrumentation.dimension)
        # keep track of what is instrumented (but "how" is probably too long/complex)
        instrumented = [f"arg{k}" if name is None else name for k, name in enumerate(self.instrumentation.names)
                        if not isinstance(self.instrumentation.instruments[k], _Constant)]
        name = function.__name__ if hasattr(function, "__name__") else function.__class__.__name__
        self._descriptors.update(name=name, instrumented=",".join(instrumented))
        self._function = function
        self.last_call_args: Optional[Tuple[Any, ...]] = None
        self.last_call_kwargs: Optional[Dict[str, Any]] = None

    def convert_to_arguments(self, data: np.ndarray, deterministic: bool = True) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
        """Get the arguments and keyword arguments corresponding to the data

        Parameters
        ----------
        data: np.ndarray
            input data
        deterministic: bool
            whether to process the data deterministically (some Variables such as SoftmaxCategorical are stochastic).
            If True, the output is the most likely output.
        """
        return self.instrumentation.data_to_arguments(data, deterministic=deterministic)

    def convert_to_data(self, *args: Any, **kwargs: Any) -> ArrayLike:
        return self.instrumentation.arguments_to_data(*args, **kwargs)

    def oracle_call(self, x: np.ndarray) -> Any:
        self.last_call_args, self.last_call_kwargs = self.convert_to_arguments(x, deterministic=False)
        return self._function(*self.last_call_args, **self.last_call_kwargs)

    def __call__(self, x: np.ndarray) -> Any:
        # BaseFunction __call__ method should generally not be overriden,
        # but here that would mess up with typing, and I would rather not constrain
        # user to return only floats.
        x = self.transform(x)
        return self.oracle_call(x)

    def get_summary(self, data: np.ndarray) -> Any:  # probably impractical for large arrays
        """Prints the summary corresponding to the provided data
        """
        strings = []
        names = self.instrumentation.names
        instruments = self.instrumentation.instruments
        splitted_data = utils.split_data(data, instruments)
        for k, (name, var, d) in enumerate(zip(names, instruments, splitted_data)):
            if not isinstance(var, _Constant):
                explanation = var.get_summary(d)
                sname = f"arg #{k + 1}" if name is None else f'kwarg "{name}"'
                strings.append(f"{sname}: {explanation}")
        return " - " + "\n - ".join(strings)
