# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union, Sequence, Any, Type
from functools import reduce
import operator
import warnings
import numpy as np
from nevergrad.parametrization import transforms
from nevergrad.parametrization import parameter as p
from nevergrad.common.typetools import ArrayLike
from .core import ArgsKwargs
from .core import Variable as Variable
from .core import VariableLayer
# pylint: disable=unused-argument,too-many-ancestors


__all__ = ["SoftmaxCategorical", "UnorderedDiscrete", "OrderedDiscrete", "Gaussian", "Array", "Scalar"]


def wrap_arg(arg: Any) -> ArgsKwargs:
    """Wrap a unique arg into args and kwargs
    """
    return (arg,), {}


class SoftmaxCategorical(VariableLayer):
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
        super().__init__(p.Choice(choices=possibilities, deterministic=deterministic))


class UnorderedDiscrete(VariableLayer):
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
        super().__init__(p.TransitionChoice(choices=possibilities))


# The ordered discrete variables are a special case of unordered discrete.
# Basically they are represented the exact same way, as Gaussian quantiles:
# the i^th (i=0,...,i=k-1) possible value of a discrete variable with k values is represented by the quantile
# (i+.5)/k of the standard normal distribution.
# The optimization algorithm can check the instrumentation to know which kind of data this is.


class OrderedDiscrete(UnorderedDiscrete):
    pass


class Gaussian(VariableLayer):
    """Gaussian variable with a mean and a standard deviation, and
    possibly a shape (when using directly in Python)
    The output will simply be mean + std * data
    """

    def __init__(self, mean: float, std: float, shape: Optional[Sequence[int]] = None) -> None:
        param = p.Scalar(init=mean) if shape is None else p.Array(init=mean * np.ones(shape))
        super().__init__(param.set_mutation(sigma=std))


class _Constant(Variable):
    """Fake variable so that constant variables can fit into the
    pipeline.
    """

    def __init__(self, value: Any) -> None:
        super().__init__()
        self.value = value
        self._specs.update(dimension=0, name=str(value))

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


class ArrayOld(Variable):
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
        dim = reduce(operator.mul, self.shape, 1)  # muuuch faster than numpy version (which converts to array)
        self._specs.update(dimension=dim)

    @property
    def name(self) -> str:
        if self._specs.name is not None:
            return self._specs.name
        # dynamic naming
        dims = str(self.shape)[1:-1].rstrip(",").replace(" ", "")
        transf = "" if not self.transforms else (",[" + ",".join(t.name for t in self.transforms) + "]")
        fl = {None: "", int: "i", float: "f"}[self._dtype]
        return f"A({dims}{transf}){fl}"

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

    def asfloat(self) -> 'Array':
        warnings.warn('Please use "asscalar" instead of "asfloat"', DeprecationWarning)
        return self.asscalar()

    def asscalar(self, dtype: Type[Union[float, int]] = float) -> 'ArrayOld':
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
        self._specs.update(continuous=dtype == float)
        self._dtype = dtype
        return self

    def with_transform(self, transform: transforms.Transform) -> 'ArrayOld':
        self.transforms.append(transform)
        return self

    def exponentiated(self, base: float, coeff: float) -> 'ArrayOld':
        """Exponentiation transform base ** (coeff * x)
        This can for instance be used for to get a logarithmicly distruted values 10**(-[1, 2, 3]).

        Parameters
        ----------
        base: float
        coeff: float
        """
        return self.with_transform(transforms.Exponentiate(base=base, coeff=coeff))

    def affined(self, a: float, b: float = 0.) -> 'ArrayOld':
        """Affine transform a * x + b

        Parameters
        ----------
        a: float
        b: float
        """
        return self.with_transform(transforms.Affine(a=a, b=b))

    def bounded(self, a_min: Optional[float] = None, a_max: Optional[float] = None, transform: str = "arctan") -> 'ArrayOld':
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


class Array(VariableLayer):
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
        param = p.Array(shape=dims) if self.__class__.__name__ == "Array" else p.Scalar()
        super().__init__(param)

    def asfloat(self) -> 'Array':
        raise Exception("Use Scalar class directly")

    def asscalar(self, dtype: Type[Union[float, int]] = float) -> 'Array':
        raise Exception("Use Scalar class directly")

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
        if dtype == int:
            self[0][0].set_integer_casting()


class ScalarOld(ArrayOld):
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


class Log(Variable):
    """Implements a log distributed variable, mapped to [-width / 2, width / 2] in the optimization "data" space
     with clipping on the boundaries.

    Parameters
    ----------
    a_min: float
        minimal value
    a_max: float
        maximal value
    width: float
        the width of the data space the bounds are mapped to. The width controls the mutation speed,
        since in the optimization data space mutations follow a centred and reduced Gaussian distribution N(0, 1)
    dtype: type
        either int or float, the resturn type of the variable

    Note
    ----
    This is experimental, and is bound to evolve with improved instrumentation
    """

    def __init__(self, a_min: float, a_max: float, width: float = 20.0, dtype: Type[Union[float, int]] = float) -> None:
        super().__init__()
        assert a_min < a_max
        self._specs.update(dimension=1, name="Log({a_min},{a_max},{width})")
        min_log = np.log10(a_min)
        max_log = np.log10(a_max)
        b = (min_log + max_log) / 2.
        a = (max_log - min_log) / width
        self._var = ScalarOld(dtype).bounded(-width / 2, width / 2, transform="clipping").affined(a=a, b=b).exponentiated(10, coeff=1)

    def _data_to_arguments(self, data: np.ndarray, deterministic: bool = True) -> ArgsKwargs:
        return self._var._data_to_arguments(data, deterministic=deterministic)

    def _arguments_to_data(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return self._var._arguments_to_data(*args, **kwargs)

    def get_summary(self, data: ArrayLike) -> str:
        return self._var.get_summary(data)

    def __repr__(self) -> str:
        return self.name
