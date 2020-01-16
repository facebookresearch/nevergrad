# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import uuid
import copy
from typing import Any, Tuple, Optional, Dict, Set, TypeVar
import typing as tp
import numpy as np
from nevergrad.common.typetools import ArrayLike
from ..parametrization.utils import Descriptors
from ..parametrization import parameter as p
# pylint: disable=no-value-for-parameter,too-many-ancestors, too-many-instance-attributes

ArgsKwargs = Tuple[Tuple[Any, ...], Dict[str, Any]]
T = TypeVar('T', bound="Variable")


class VarSpecs:

    # pylint: disable=too-many-arguments, unused-argument
    def __init__(self) -> None:
        self.dimension = -1
        self.nargs = 1
        self.kwargs_keys: Set[str] = set()
        self.continuous = True
        self.noisy = False
        self.name: Optional[str] = None

    def update(self,
               dimension: Optional[int] = None,
               nargs: Optional[int] = None,
               kwargs_keys: Optional[Set[str]] = None,
               continuous: Optional[bool] = None,
               noisy: Optional[bool] = None,
               name: Optional[str] = None
               ) -> None:
        for key, value in locals().items():
            if key != "self" and value is not None:
                setattr(self, key, value)


def _default_checker(*args: Any, **kwargs: Any) -> bool:  # pylint: disable=unused-argument
    return True


class Variable(p.Instrumentation):

    def __init__(self) -> None:
        super().__init__()
        self._specs = VarSpecs()
        # compatibility
        self._data: tp.Optional[np.ndarray] = None
        self._value: tp.Optional[ArgsKwargs] = None

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            self._data = np.zeros((self.dimension,))
        return self._data

    def _set_random_state(self, random_state: np.random.RandomState) -> None:
        self._random_state = random_state

    def _get_name(self) -> str:
        if self._specs.name is not None:
            return self._specs.name
        return repr(self)

    def with_name(self: T, name: str) -> T:
        """Sets a name and return the current instrumentation (for chaining)
        """
        return self.set_name(name)

    def copy(self: T) -> T:  # TODO, use deepcopy directly in the code if it works?
        """Return a new instrumentation with the same variable and same name
        (but a different random state)
        """
        instru = copy.deepcopy(self)
        instru._random_state = None
        return instru

    @property
    def dimension(self) -> int:
        return self._specs.dimension

    @property
    def nargs(self) -> int:
        return self._specs.nargs

    @property
    def kwargs_keys(self) -> Set[str]:
        return self._specs.kwargs_keys

    @property
    def continuous(self) -> bool:
        return self._specs.continuous

    @property
    def noisy(self) -> bool:
        return self._specs.noisy

    def arguments_to_data(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Converts args and kwargs into data in np.ndarray format
        """
        if len(args) != self.nargs:
            raise TypeError(f"Expected {self.nargs} arguments ({len(args)} given: {args})")
        if self.kwargs_keys != set(kwargs.keys()):
            raise TypeError(f"Expected arguments {self.kwargs_keys} ({set(kwargs.keys())} given: {kwargs})")
        return self._arguments_to_data(*args, **kwargs)

    def _arguments_to_data(self, *args: Any, **kwargs: Any) -> np.ndarray:
        raise RuntimeError(f"arguments_to_data is not defined for {self.__class__.__name__}")

    def data_to_arguments(self, data: ArrayLike, deterministic: bool = False) -> ArgsKwargs:
        """Converts data to arguments

        Parameters
        ----------
        data: ArrayLike (list/tuple of floats, np.ndarray)
            the data in the optimization space
        deterministic: bool
            whether the conversion should be deterministic (some variables can be stochastic, if deterministic=True
            the most likely output will be used)

        Returns
        -------
        args: Tuple[Any]
            the positional arguments corresponding to the instance initialization positional arguments
        kwargs: Dict[str, Any]
            the keyword arguments corresponding to the instance initialization keyword arguments
        """
        # trigger random_state creation (may require to be propagated to sub-variables
        assert self.random_state is not None
        array = np.array(data, copy=False)
        if array.shape != (self.dimension,):
            raise ValueError(f"Unexpected shape {array.shape} of {array} for {self} with dimension {self.dimension}")
        return self._data_to_arguments(array, deterministic)

    def _data_to_arguments(self, data: np.ndarray, deterministic: bool) -> ArgsKwargs:
        raise NotImplementedError

    def get_summary(self, data: ArrayLike) -> str:  # pylint: disable=unused-argument
        output = self.data_to_arguments(np.array(data, copy=False), deterministic=True)
        return f"Value {output[0][0]}, from data: {data}"

    # compatibility
    @property
    def value(self) -> tp.Any:
        if self._value is None:
            self._value = self.data_to_arguments(self.data)
        return self._value[0][0]

    @value.setter
    def value(self, value: tp.Any) -> None:
        self._value = (value,), {}
        self._data = self.arguments_to_data(value)

    @property
    def args(self) -> tp.Tuple[tp.Any, ...]:
        return (self.value,)

    @property
    def kwargs(self) -> tp.Dict[str, tp.Any]:
        return {}

    def _internal_get_standardized_data(self: T, reference: T) -> np.ndarray:  # pylint: disable=unused-argument
        return self.data - reference.data  # type: ignore

    def _internal_set_standardized_data(self: T, data: np.ndarray, reference: T, deterministic: bool = False) -> None:
        self._data = data + reference.data
        self._value = self.data_to_arguments(self.data, deterministic=deterministic)

    def _internal_spawn_child(self: T) -> T:
        child = copy.deepcopy(self)
        child._frozen = False
        child.uid = uuid.uuid4().hex
        child.parents_uids = []
        return child

    def _compute_descriptors(self) -> Descriptors:
        return Descriptors(continuous=self.continuous, deterministic=not self.noisy)

    def mutate(self) -> None:
        raise p.NotSupportedError("Please port your code to new parametrization")

    def recombine(self: T, *others: T) -> None:  # type: ignore
        raise p.NotSupportedError("Please port your code to new parametrization")
