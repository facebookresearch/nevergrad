# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import bisect
from enum import Enum
import numpy as np
from nevergrad.common import errors
import nevergrad.common.typing as tp


L = tp.TypeVar("L", bound="Layered")
F = tp.TypeVar("F", bound="Filterable")
In = tp.TypeVar("In")
Out = tp.TypeVar("Out")


class Level(Enum):
    """Lower level is deeper in the structure"""

    ROOT = 0
    OPERATION = 10

    # final
    INTEGER_CASTING = 800
    ARRAY_CASTING = 900
    SCALAR_CASTING = 950
    CONSTRAINT = 1000  # must be the last layer


class Layered:
    """Hidden API for overriding/modifying the behavior of a Parameter,
    which is itself a Layered object.

    Layers can be added and will be ordered depending on their level
    """

    _LAYER_LEVEL = Level.OPERATION  # this provides an order for the layers

    def __init__(self) -> None:
        self._layers = [self]
        self._layer_index = 0
        self._name: tp.Optional[str] = None

    def add_layer(self: L, other: "Layered") -> L:
        """Adds a layer which will modify the object behavior"""
        if self is not self._layers[0] or self._LAYER_LEVEL != Level.ROOT:
            raise errors.NevergradRuntimeError("Layers can only be added from the root.")
        if len(other._layers) > 1:
            raise errors.NevergradRuntimeError("Cannot append multiple layers at once")
        if other._LAYER_LEVEL.value >= self._layers[-1]._LAYER_LEVEL.value:
            other._layer_index = len(self._layers)
            self._layers.append(other)
        else:
            levels = [x._LAYER_LEVEL.value for x in self._layers]
            ind = bisect.bisect_right(levels, other._LAYER_LEVEL.value)
            self._layers.insert(ind, other)
            for k, x in enumerate(self._layers):
                x._layer_index = k
        other._layers = self._layers
        return self

    def _call_deeper(self, name: str, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        if self._layers[self._layer_index] is not self:
            layers = [f"{l.name}({l._layer_index})" for l in self._layers]
            raise errors.NevergradRuntimeError(
                "Layer indexing has changed for an unknown reason. Please open an issue:\n"
                f"Caller at index {self._layer_index}: {self.name}"
                f"Layers: {layers}.\n"
            )
        if not name.startswith("_layered_"):
            raise errors.NevergradValueError("For consistency, only _layered functions can be used.")
        for layer in reversed(self._layers[: self._layer_index]):
            func = getattr(layer, name)
            if func.__func__ is not getattr(Layered, name):  # skip unecessary stack calls
                return func(*args, **kwargs)
        types = [type(x) for x in self._layers]
        raise errors.NevergradNotImplementedError(f"No implementation for {name} on layers: {types}.")
        # ALTERNATIVE (stacking all calls):
        # if not self._layer_index:  # root must have an implementation
        #    raise errors.NevergradNotImplementedError
        # return getattr(self._layers[self._layer_index - 1], name)(*args, **kwargs)

    def _layered_get_value(self) -> tp.Any:
        return self._call_deeper("_layered_get_value")

    def _layered_set_value(self, value: tp.Any) -> tp.Any:
        return self._call_deeper("_layered_set_value", value)

    def _layered_del_value(self) -> None:
        pass  # called independently on each layer

    def _layered_sample(self) -> "Layered":
        return self._call_deeper("_layered_sample")  # type: ignore

    @property
    def random_state(self) -> np.random.RandomState:
        return self._layers[0].random_state  # use the root random state

    def copy(self: L) -> L:
        """Creates a new unattached layer with the same behavior"""
        new = copy.copy(self)
        new._layers = [new]
        new._layer_index = 0
        if not self._layer_index:  # attach sublayers if root
            for layer in self._layers[1:]:
                new.add_layer(layer.copy())
        return new

    # naming capacity

    def _get_name(self) -> str:
        """Internal implementation of parameter name. This should be value independant, and should not account
        for internal/model parameters.
        """
        return self.__class__.__name__

    def __repr__(self) -> str:
        return self.name

    @property
    def name(self) -> str:
        """Name of the parameter
        This is used to keep track of how this Parameter is configured (included through internal/model parameters),
        mostly for reproducibility A default version is always provided, but can be overriden directly
        through the attribute, or through the set_name method (which allows chaining).
        """
        if self._name is not None:
            return self._name
        return self._get_name()

    @name.setter
    def name(self, name: str) -> None:
        self.set_name(name)  # with_name allows chaining

    def set_name(self: L, name: str) -> L:
        """Sets a name and return the current instrumentation (for chaining)

        Parameters
        ----------
        name: str
            new name to use to represent the Parameter
        """
        self._name = name
        return self


class ValueProperty(tp.Generic[In, Out]):
    """Typed property (descriptor) object so that the value attribute of
    Parameter objects fetches _layered_get_value and _layered_set_value methods
    """

    # This uses the descriptor protocol, like a property:
    # See https://docs.python.org/3/howto/descriptor.html
    #
    # Basically parameter.value calls parameter.value.__get__
    # and then parameter._layered_get_value
    def __init__(self) -> None:
        self.__doc__ = """Value of the Parameter, which should be sent to the function
        to optimize.

        Example
        -------
        >>> ng.p.Array(shape=(2,)).value
        array([0., 0.])
        """

    def __get__(self, obj: Layered, objtype: tp.Optional[tp.Type[object]] = None) -> Out:
        return obj._layers[-1]._layered_get_value()  # type: ignore

    def __set__(self, obj: Layered, value: In) -> None:
        self.__delete__(obj)
        obj._layers[-1]._layered_set_value(value)

    def __delete__(self, obj: Layered) -> None:
        for layer in obj._layers:
            layer._layered_del_value()


# Basic data layers


class _ScalarCasting(Layered):
    """Cast Array as a scalar"""

    _LAYER_LEVEL = Level.SCALAR_CASTING

    def _layered_get_value(self) -> float:
        out = super()._layered_get_value()  # pulls from previous layer
        if not isinstance(out, np.ndarray) or not out.size == 1:
            raise errors.NevergradRuntimeError("Scalar casting can only be applied to size=1 Data parameters")
        integer = np.issubdtype(out.dtype, np.integer)
        out = (int if integer else float)(out[0])
        return out  # type: ignore

    def _layered_set_value(self, value: tp.Any) -> None:
        if not isinstance(value, (float, int, np.float, np.int)):
            raise TypeError(f"Received a {type(value)} in place of a scalar (float, int)")
        super()._layered_set_value(np.array([value], dtype=float))


class ArrayCasting(Layered):
    """Cast inputs of type tuple/list etc to array"""

    _LAYER_LEVEL = Level.ARRAY_CASTING

    def _layered_set_value(self, value: tp.ArrayLike) -> None:
        if not isinstance(value, (np.ndarray, tuple, list)):
            raise TypeError(f"Received a {type(value)} in place of a np.ndarray/tuple/list")
        super()._layered_set_value(np.asarray(value))


class Filterable:
    @classmethod
    def filter_from(cls: tp.Type[F], parameter: Layered) -> tp.List[F]:
        return [x for x in parameter._layers if isinstance(x, cls)]  # type: ignore


class Int(Layered, Filterable):
    """Cast Data as integer (or integer array)

    Parameters
    ----------
    deterministic: bool
        if True, the data is rounded to the closest integer, if False, both surrounded
        integers can be sampled inversely proportionally to how close the actual value
        is from the integers.

    Example
    -------
    0.2 is cast to 0 in deterministic mode, and either 0 (80% chance) or 1 (20% chance) in
    non-deterministic mode
    """

    _LAYER_LEVEL = Level.INTEGER_CASTING

    def __init__(self, deterministic: bool = True) -> None:
        super().__init__()
        self.arity: tp.Optional[int] = None
        self.ordered = True
        self.deterministic = deterministic
        self._cache: tp.Optional[np.ndarray] = None

    def _get_name(self) -> str:
        tag = "" if self.deterministic else "{rand}"
        return self.__class__.__name__ + tag

    def _layered_get_value(self) -> np.ndarray:
        if self._cache is not None:
            return self._cache
        bounds = self._layers[0].bounds  # type: ignore
        out = super()._layered_get_value()
        if not self.deterministic:
            out += self.random_state.rand(*out.shape) - 0.5
        out = np.round(out).astype(int)
        # make sure rounding does not reach beyond the bounds
        eps = 1e-12
        if bounds[0] is not None:
            out = np.maximum(int(np.round(bounds[0] + 0.5 - eps)), out)
        if bounds[1] is not None:
            out = np.minimum(int(np.round(bounds[1] - 0.5 + eps)), out)
        # return out
        self._cache = out
        return self._cache

    def _layered_del_value(self) -> None:
        self._cache = None  # clear cache!
