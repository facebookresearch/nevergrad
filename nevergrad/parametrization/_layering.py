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
X = tp.TypeVar("X")


class Level(Enum):
    """Lower level is deeper in the structure"""

    ROOT = 0
    OPERATION = 10
    CASTING = 90

    # final
    ARRAY_CASTING = 900
    INTEGER_CASTING = 1000  # must be the last layer


class Layered:
    """Hidden API for overriding/modifying the behavior of a Parameter,
    which is itself a Layered object.

    Layers can be added and will be ordered depending on their level
    """

    _LAYER_LEVEL = Level.OPERATION

    def __init__(self) -> None:
        self._layers = [self]
        self._index = 0
        self._name: tp.Optional[str] = None

    def _get_layer_index(self) -> int:
        if self._layers[self._index] is not self:
            layers = [f"{l.name}({l._index})" for l in self._layers]
            raise errors.NevergradRuntimeError(
                "Layer indexing has changed for an unknown reason. Please open an issue:\n"
                f"Caller at index {self._index}: {self.name}"
                f"Layers: {layers}.\n"
            )
        return self._index

    def _get_value(self) -> tp.Any:
        index = self._get_layer_index()
        if not index:  # root must have an implementation
            raise NotImplementedError
        return self._layers[index - 1]._get_value()

    def _set_value(self, value: tp.Any) -> tp.Any:
        index = self._get_layer_index()
        if not index:  # root must have an implementation
            raise NotImplementedError
        self._layers[index - 1]._set_value(value)

    def _del_value(self) -> tp.Any:
        pass

    def add_layer(self: L, other: "Layered") -> L:
        """Adds a layer which will modify the object behavior"""
        if self is not self._layers[0] or self._LAYER_LEVEL != Level.ROOT:
            raise errors.NevergradRuntimeError("Layers can only be added from the root.")
        if len(other._layers) > 1:
            raise errors.NevergradRuntimeError("Cannot append multiple layers at once")
        if other._LAYER_LEVEL.value >= self._layers[-1]._LAYER_LEVEL.value:
            other._index = len(self._layers)
            self._layers.append(other)
        else:
            levels = [x._LAYER_LEVEL.value for x in self._layers]
            ind = bisect.bisect_right(levels, other._LAYER_LEVEL.value)
            self._layers.insert(ind, other)
            for k, x in enumerate(self._layers):
                x._index = k
        other._layers = self._layers
        return self

    def copy(self: L) -> L:
        """Creates a new unattached layer with the same behavior"""
        new = copy.copy(self)
        new._layers = [new]
        new._index = 0
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


class ValueProperty(tp.Generic[X]):
    """Typed property (descriptor) object so that the value attribute of
    Parameter objects fetches _get_value and _set_value methods
    """

    # This uses the descriptor protocol, like a property:
    # See https://docs.python.org/3/howto/descriptor.html
    #
    # Basically parameter.value calls parameter.value.__get__
    # and then parameter._get_value
    def __init__(self) -> None:
        self.__doc__ = """Value of the Parameter, which should be sent to the function
        to optimize.

        Example
        -------
        >>> ng.p.Array(shape=(2,)).value
        array([0., 0.])
        """

    def __get__(self, obj: Layered, objtype: tp.Optional[tp.Type[object]] = None) -> X:
        return obj._layers[-1]._get_value()  # type: ignore

    def __set__(self, obj: Layered, value: X) -> None:
        obj._layers[-1]._set_value(value)

    def __delete__(self, obj: Layered) -> None:
        for layer in obj._layers:
            layer._del_value()


# Basic data layers


class _ScalarCasting(Layered):
    """Cast Array as a scalar"""

    _LAYER_LEVEL = Level.INTEGER_CASTING

    def _get_value(self) -> float:
        out = super()._get_value()  # pulls from previous layer
        if not isinstance(out, np.ndarray) or not out.size == 1:
            raise errors.NevergradRuntimeError("Scalar casting can only be applied to size=1 Data parameters")
        integer = np.issubdtype(out.dtype, np.integer)
        out = (int if integer else float)(out[0])
        return out  # type: ignore

    def _set_value(self, value: tp.Any) -> None:
        if not isinstance(value, (float, int, np.float, np.int)):
            raise TypeError(f"Received a {type(value)} in place of a scalar (float, int)")
        super()._set_value(np.array([value], dtype=float))


class ArrayCasting(Layered):
    """Cast inputs of type tuple/list etc to array"""

    _LAYER_LEVEL = Level.ARRAY_CASTING

    def _set_value(self, value: tp.ArrayLike) -> None:
        if not isinstance(value, (np.ndarray, tuple, list)):
            raise TypeError(f"Received a {type(value)} in place of a np.ndarray/tuple/list")
        super()._set_value(np.asarray(value))


class IntegerCasting(Layered):
    """Cast Data as integer (or integer array)"""

    _LAYER_LEVEL = Level.CASTING

    def _get_value(self) -> np.ndarray:
        return np.round(super()._get_value()).astype(int)  # type: ignore


class Modulo(Layered):
    """Cast Data as integer (or integer array)"""

    _LAYER_LEVEL = Level.OPERATION

    def __init__(self, module: tp.Any) -> None:
        super().__init__()
        if not isinstance(module, (np.ndarray, np.float, np.int, float, int)):
            raise TypeError(f"Unsupported type {type(module)} for module")
        self._module = module

    def _get_value(self) -> np.ndarray:
        return super()._get_value() % self._module  # type: ignore

    def _set_value(self, value: np.ndarray) -> None:
        current = super()._get_value()
        super()._set_value(current - (current % self._module) + value)
