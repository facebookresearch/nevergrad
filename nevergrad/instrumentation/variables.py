# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, TypeVar, Union, Sequence, Any, Type
import warnings
import numpy as np
from . import discretization
from ..common.typetools import ArrayLike
from . import transforms
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
        assert len(possibilities) > 1, ("Variable needs at least 2 values to choose from (constant values can be directly used as input "
                                        "for the Instrumentation intialization")

    @property
    def continuous(self) -> bool:
        return not self.deterministic

    @property
    def noisy(self) -> bool:
        return not self.deterministic

    @property
    def dimension(self) -> int:
        return len(self.possibilities)

    def data_to_argument(self, data: ArrayLike, deterministic: bool = False) -> X:
        assert len(data) == len(self.possibilities)
        deterministic = deterministic | self.deterministic
        index = int(discretization.softmax_discretization(data, len(self.possibilities), deterministic=deterministic)[0])
        return self.possibilities[index]

    def argument_to_data(self, arg: X) -> ArrayLike:
        assert arg in self.possibilities, f'{arg} not in allowed values: {self.possibilities}'
        return discretization.inverse_softmax_discretization(self.possibilities.index(arg), len(self.possibilities))

    def get_summary(self, data: ArrayLike) -> str:
        output = self.data_to_argument(data, deterministic=True)
        probas = discretization.softmax_probas(np.array(data, copy=False))
        proba_str = ", ".join([f'"{s}": {round(100 * p)}%' for s, p in zip(self.possibilities, probas)])
        return f"Value {output}, from data: {data} yielding probas: {proba_str}"

    def _short_repr(self) -> str:
        return "SC({}|{})".format(",".join([str(x) for x in self.possibilities]), int(self.deterministic))


class OrderedDiscrete(utils.Variable[X]):
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

    def __init__(self, possibilities: List[X]) -> None:
        self.possibilities = list(possibilities)
        assert len(possibilities) > 1, ("Variable needs at least 2 values to choose from (constant values can be directly used as input "
                                        "for the Instrumentation intialization")

    @property
    def continuous(self) -> bool:
        return False

    @property
    def dimension(self) -> int:
        return 1

    def data_to_argument(self, data: ArrayLike, deterministic: bool = False) -> X:  # pylint: disable=arguments-differ, unused-argument
        assert len(data) == 1
        index = discretization.threshold_discretization(data, arity=len(self.possibilities))[0]
        return self.possibilities[index]

    def argument_to_data(self, arg: X) -> ArrayLike:
        assert arg in self.possibilities, f'{arg} not in allowed values: {self.possibilities}'
        index = self.possibilities.index(arg)
        return discretization.inverse_threshold_discretization([index], len(self.possibilities))

    def _short_repr(self) -> str:
        return "OD({})".format(",".join([str(x) for x in self.possibilities]))


Y = Union[int, float, np.ndarray]


class Gaussian(utils.Variable[Y]):
    """Gaussian variable with a mean and a standard deviation, and
    possibly a shape (when using directly in Python)
    The output will simply be mean + std * data
    """

    def __init__(self, mean: float, std: float, shape: Optional[Sequence[int]] = None) -> None:
        self.mean = mean
        self.std = std
        self.shape = shape

    @property
    def dimension(self) -> int:
        return 1 if self.shape is None else int(np.prod(self.shape))

    def data_to_argument(self, data: ArrayLike, deterministic: bool = True) -> Y:  # pylint: disable=unused-argument
        assert len(data) == self.dimension
        x = data[0] if self.shape is None else np.reshape(data, self.shape)
        return self.std * x + self.mean

    def argument_to_data(self, arg: Y) -> ArrayLike:
        return [(arg - self.mean) / self.std]

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

    def data_to_argument(self, data: ArrayLike, deterministic: bool = False) -> X:  # pylint: disable=unused-argument
        return self.value

    def argument_to_data(self, arg: X) -> ArrayLike:
        assert arg == self.value, f'{arg} != {self.value}'
        return []

    def get_summary(self, data: ArrayLike) -> str:
        raise RuntimeError("Constant summary should not be called")

    def _short_repr(self) -> str:
        return f"{self.value}"


class Array(utils.Variable[Y]):
    """Array variable of a given shape, on which several transforms can be applied.

    Parameters
    ----------
    *dims: int
        dimensions of the array (elements of shape)

    Note
    ----
    Interesting methods (which can be chained):
    - asscalar(): converts the array into a float or int (only for arrays with 1 element)
    - with_transform(transform): apply a transform to the array
    - affined(a, b): applies a*x+b
    - bounded(a_min, a_max, transform="tanh"): applies a transform ("tanh" or "arctan")
      so that output values are in range [a_min, a_max]
    - exponentiated(base, coeff): applies base**(coeff * x)
    """

    def __init__(self, *dims: int) -> None:
        self.transforms: List[Any] = []
        self.shape = tuple(dims)
        self._scalar_type: Optional[Type[Union[float, int]]] = None

    @property
    def dimension(self) -> int:
        return int(np.prod(self.shape))

    @property
    def continuous(self) -> bool:
        return self._scalar_type != int

    def data_to_argument(self, data: ArrayLike, deterministic: bool = False) -> Y:  # pylint: disable=unused-argument
        assert len(data) == self.dimension
        array = np.array(data, copy=False)
        for transf in self.transforms:
            array = transf.forward(array)
        if self._scalar_type is not None:
            return self._scalar_type(array[0] if self._scalar_type != int else round(array[0]))
        return array.reshape(self.shape)

    def argument_to_data(self, arg: Y) -> np.ndarray:
        if self._scalar_type is not None:
            output = np.array([arg], dtype=float)
        else:
            output = np.array(arg, copy=False).ravel()
        for transf in reversed(self.transforms):
            output = transf.backward(output)
        return output

    def _short_repr(self) -> str:
        dims = ",".join(str(d) for d in self.shape)
        transf = "" if not self.transforms else (",[" + ",".join(f"{t:short}" for t in self.transforms) + "]")
        fl = {None: "", int: "i", float: "f"}[self._scalar_type]
        return f"A({dims}{transf}){fl}"

    def asfloat(self) -> 'Array':
        warnings.warn('Please use "asscalar" instead of "asfloat"', DeprecationWarning)
        return self.asscalar()

    def asscalar(self, scalar_type: Type[Union[float, int]] = float) -> 'Array':
        """Converts the array into a scalar

        Parameter
        ---------
        scalar_type: type
            either int or float

        Note
        ----
        This method can only be called on size 1 arrays
        """
        if self._scalar_type is not None:
            raise RuntimeError('"asscalar" must only be called once')
        if self.dimension != 1:
            raise RuntimeError("Only Arrays with 1 element can be cast to float")
        if scalar_type not in [float, int]:
            raise ValueError('"scalar_type" should be either float or int')
        self._scalar_type = scalar_type
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
        a_max: float or None
        transform: str
            "clipping", "tanh" or "arctan

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
