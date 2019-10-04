# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union, Sequence, Any, Type
from functools import reduce
import operator
import warnings
import numpy as np
from . import discretization
from ..common.typetools import ArrayLike
from . import transforms
from .core2 import ArgsKwargs
from .core2 import Variable
# pylint: disable=unused-argument


__all__ = ["SoftmaxCategorical", "OrderedDiscrete", "Gaussian", "Array", "Scalar"]


def wrap_arg(arg: Any) -> ArgsKwargs:
    """Wrap a unique arg into args and kwargs
    """
    return (arg,), {}


class SoftmaxCategorical(Variable):
    """Discrete set of n values transformed to a n-dim continuous variable.
    Each of the dimension encodes a weight for a value, and the softmax of weights
    provide probabilities for each possible value. A random value is sampled from
    this distribution.

    Parameters
    ----------
    possibilities: list
        a list of possible values for the variable.

    Note
    ----
    Since the chosen value is drawn randomly, the use of this variable makes deterministic
    functions become stochastic, hence "adding noise"
    """

    def __init__(self, possibilities: List[Any], deterministic: bool = False) -> None:
        super().__init__()
        self.deterministic = deterministic
        self.possibilities = list(possibilities)
        self._specs.update(dimension=len(self.possibilities), continuous=not self.deterministic, noisy=not self.deterministic)
        assert len(possibilities) > 1, ("Variable needs at least 2 values to choose from (constant values can be directly used as input "
                                        "for the Instrumentation intialization")

    def _data_to_arguments(self, data: np.ndarray, deterministic: bool = True) -> ArgsKwargs:
        random = False if deterministic or self.deterministic else self.random_state
        index = int(discretization.softmax_discretization(data, len(self.possibilities), random=random)[0])
        return wrap_arg(self.possibilities[index])

    def _arguments_to_data(self, *args: Any, **kwargs: Any) -> np.ndarray:
        arg = args[0]
        assert arg in self.possibilities, f'{arg} not in allowed values: {self.possibilities}'
        out = discretization.inverse_softmax_discretization(self.possibilities.index(arg), len(self.possibilities))
        return np.array(out, copy=False)

    def get_summary(self, data: np.ndarray) -> str:
        output = self.data_to_arguments(data, deterministic=True)
        probas = discretization.softmax_probas(np.array(data, copy=False))
        proba_str = ", ".join([f'"{s}": {round(100 * p)}%' for s, p in zip(self.possibilities, probas)])
        return f"Value {output}, from data: {data} yielding probas: {proba_str}"

    def _short_repr(self) -> str:
        return "SC({}|{})".format(",".join([str(x) for x in self.possibilities]), int(self.deterministic))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.possibilities}, {self.deterministic})"


class OrderedDiscrete(Variable):
    """Discrete list of n values transformed to a 1-dim discontinuous variable.
    A gaussian input yields a uniform distribution on the list of variables.

    Parameters
    ----------
    possibilities: list
        a list of possible values for the variable.

    Note
    ----
    The variables are assumed to be ordered.
    """

    def __init__(self, possibilities: List[Any]) -> None:
        super().__init__()
        self.possibilities = list(possibilities)
        self._specs.update(continuous=False, dimension=1)
        assert len(possibilities) > 1, ("Variable needs at least 2 values to choose from (constant values can be directly used as input "
                                        "for the Instrumentation intialization")

    def _data_to_arguments(self, data: np.ndarray, deterministic: bool = True) -> ArgsKwargs:
        index = discretization.threshold_discretization(data, arity=len(self.possibilities))[0]
        return wrap_arg(self.possibilities[index])

    def _arguments_to_data(self, *args: Any, **kwargs: Any) -> np.ndarray:
        arg = args[0]
        assert arg in self.possibilities, f'{arg} not in allowed values: {self.possibilities}'
        index = self.possibilities.index(arg)
        out = discretization.inverse_threshold_discretization([index], len(self.possibilities))
        return np.array(out, copy=False)

    def _short_repr(self) -> str:
        return "OD({})".format(",".join([str(x) for x in self.possibilities]))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.possibilities})"


class Gaussian(Variable):
    """Gaussian variable with a mean and a standard deviation, and
    possibly a shape (when using directly in Python)
    The output will simply be mean + std * data
    """

    def __init__(self, mean: float, std: float, shape: Optional[Sequence[int]] = None) -> None:
        super().__init__()
        self.mean = mean
        self.std = std
        self.shape = shape
        self._specs.update(dimension=1 if self.shape is None else int(np.prod(self.shape)))

    def _data_to_arguments(self, data: np.ndarray, deterministic: bool = True) -> ArgsKwargs:
        x = data[0] if self.shape is None else np.reshape(data, self.shape)
        return wrap_arg(self.std * x + self.mean)

    def _arguments_to_data(self, *args: Any, **kwargs: Any) -> np.ndarray:
        out = (args[0] - self.mean) / self.std
        return np.array([out]) if self.shape is None else out  # type: ignore

    def _short_repr(self) -> str:
        return f"G({self.mean},{self.std})"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.mean},{self.std})"


class _Constant(Variable):
    """Fake variable so that constant variables can fit into the
    pipeline.
    """

    def __init__(self, value: Any) -> None:
        super().__init__()
        self.value = value
        self._specs.update(dimension=0)

    @classmethod
    def convert_non_instrument(cls, x: Any) -> Variable:
        return x if isinstance(x, Variable) else cls(x)

    def _data_to_arguments(self, data: np.ndarray, deterministic: bool = True) -> ArgsKwargs:
        return wrap_arg(self.value)

    def _arguments_to_data(self, *args: Any, **kwargs: Any) -> np.ndarray:
        assert args[0] == self.value, f'{args[0]} != {self.value}'
        return np.array([])

    def get_summary(self, data: ArrayLike) -> str:
        raise RuntimeError("Constant summary should not be called")

    def _short_repr(self) -> str:
        return f"{self.value}"

    def __repr__(self) -> str:
        return f"Constant({self.value})"


class Array(Variable):
    """Array variable of a given shape, on which several transforms can be applied.

    Parameters
    ----------
    *dims: int
        dimensions of the array (elements of shape)

    Note
    ----
    Interesting methods (which can be chained):

    - asscalar(): converts the array into a float or int (only for arrays with 1 element)
      You may also directly use `Scalar` the scalar object instead.
    - with_transform(transform): apply a transform to the array
    - affined(a, b): applies a*x+b
    - bounded(a_min, a_max, transform="tanh"): applies a transform ("tanh" or "arctan")
      so that output values are in range [a_min, a_max]

    - exponentiated(base, coeff): applies base**(coeff * x)
    """

    def __init__(self, *dims: int) -> None:
        super().__init__()
        self.transforms: List[Any] = []
        self.shape = tuple(dims)
        self._dtype: Optional[Type[Union[float, int]]] = None

    @property
    def dimension(self) -> int:
        return reduce(operator.mul, self.shape, 1)  # muuuch faster than numpy version (which converts to array)

    @property
    def continuous(self) -> bool:
        return self._dtype != int

    def _data_to_arguments(self, data: np.ndarray, deterministic: bool = True) -> ArgsKwargs:
        assert len(data) == self.dimension
        array = np.array(data, copy=False)
        for transf in self.transforms:
            array = transf.forward(array)
        if self._dtype is not None:
            out = self._dtype(array[0] if self._dtype != int else round(array[0]))
        else:
            out = array.reshape(self.shape)  # type: ignore
        return wrap_arg(out)

    def _arguments_to_data(self, *args: Any, **kwargs: Any) -> np.ndarray:
        if self._dtype is not None:
            output = np.array([args[0]], dtype=float)
        else:
            output = np.array(args[0], copy=False).ravel()
        for transf in reversed(self.transforms):
            output = transf.backward(output)
        return output

    def __repr__(self) -> str:
        return f"Array({self.shape}, {self.transforms})"

    def _short_repr(self) -> str:
        dims = ",".join(str(d) for d in self.shape)
        transf = "" if not self.transforms else (",[" + ",".join(f"{t:short}" for t in self.transforms) + "]")
        fl = {None: "", int: "i", float: "f"}[self._dtype]
        return f"A({dims}{transf}){fl}"

    def asfloat(self) -> 'Array':
        warnings.warn('Please use "asscalar" instead of "asfloat"', DeprecationWarning)
        return self.asscalar()

    def asscalar(self, dtype: Type[Union[float, int]] = float) -> 'Array':
        """Converts the array into a scalar

        Parameters
        ----------
        dtype: type
            either int or float

        Note
        ----
        This method can only be called on size 1 arrays
        """
        if self._dtype is not None:
            raise RuntimeError('"asscalar" must only be called once')
        if self.dimension != 1:
            raise RuntimeError("Only Arrays with 1 element can be cast to float")
        if dtype not in [float, int]:
            raise ValueError('"dtype" should be either float or int')
        self._dtype = dtype
        return self

    def with_transform(self, transform: transforms.Transform) -> 'Array':
        self.transforms.append(transform)
        return self

    def exponentiated(self, base: float, coeff: float) -> 'Array':
        """Exponentiation transform base ** (coeff * x)
        This can for instance be used for to get a logarithmicly distruted values 10**(-[1, 2, 3]).

        Parameters
        ----------
        base: float
        coeff: float
        """
        return self.with_transform(transforms.Exponentiate(base=base, coeff=coeff))

    def affined(self, a: float, b: float = 0.) -> 'Array':
        """Affine transform a * x + b

        Parameters
        ----------
        a: float
        b: float
        """
        return self.with_transform(transforms.Affine(a=a, b=b))

    def bounded(self, a_min: Optional[float] = None, a_max: Optional[float] = None, transform: str = "arctan") -> 'Array':
        """Bounds all real values into [a_min, a_max] using a tanh transform.
        Beware, tanh goes very fast to its limits.

        Parameters
        ----------
        a_min: float or None
            minimum value
        a_max: float or None
            maximum value
        transform: str
            "clipping", "tanh" or "arctan"

        Notes
        -----
        - "tanh" reaches the boundaries really quickly, while "arctan" is much softer
        - only "clipping" accepts partial bounds (None values)
        """
        if transform not in ["tanh", "arctan", "clipping"]:
            raise ValueError("Only 'tanh', 'clipping' and 'arctan' are allowed as transform")
        if transform in ["arctan", "tanh"]:
            Transf = transforms.ArctanBound if transform == "arctan" else transforms.TanhBound
            assert a_min is not None and a_max is not None, "Only 'clipping' can be used for partial bounds"
            return self.with_transform(Transf(a_min=a_min, a_max=a_max))
        else:
            return self.with_transform(transforms.Clipping(a_min, a_max))


class Scalar(Array):
    """Scalar variable, on which several transforms can be applied.

    Parameters
    ----------
    dtype: type
        either int or float

    Note
    ----
    Interesting methods (which can be chained):

    - with_transform(transform): apply a transform to the array
    - affined(a, b): applies a*x+b
    - bounded(a_min, a_max, transform="tanh"): applies a transform ("tanh" or "arctan")
      so that output values are in range [a_min, a_max]
    - exponentiated(base, coeff): applies base**(coeff * x)
    - `Scalar(dtype)` is completely equivalent to `Array(1).asscalar(dtype)`
    """

    def __init__(self, dtype: Type[Union[float, int]] = float):
        super().__init__(1)
        self.asscalar(dtype=dtype)
