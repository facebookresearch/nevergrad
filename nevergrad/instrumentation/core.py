# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Any, Tuple, Optional, Dict, Set, TypeVar, Callable
import numpy as np
from ..common.typetools import ArrayLike

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


class Variable:

    def __init__(self) -> None:
        self._random_state: Optional[np.random.RandomState] = None  # lazy initialization
        self._specs = VarSpecs()
        self._constraint_checker = _default_checker

    def set_cheap_constraint_checker(self, func: Callable[..., bool]) -> None:
        self._constraint_checker = func

    def cheap_constraint_check(self, *args: Any, **kwargs: Any) -> bool:
        return self._constraint_checker(*args, **kwargs)

    @property
    def random_state(self) -> np.random.RandomState:
        """Random state the instrumentation and the optimizers pull from.
        It can be seeded/replaced.
        """
        if self._random_state is None:
            # use the setter, to make sure the random state is propagated to the variables
            seed = np.random.randint(2 ** 32, dtype=np.uint32)
            self._set_random_state(np.random.RandomState(seed))
        assert self._random_state is not None
        return self._random_state

    @random_state.setter
    def random_state(self, random_state: np.random.RandomState) -> None:
        self._set_random_state(random_state)

    def _set_random_state(self, random_state: np.random.RandomState) -> None:
        self._random_state = random_state

    def with_name(self: T, name: str) -> T:
        """Sets a name and return the current instrumentation (for chaining)
        """
        self._specs.update(name=name)
        return self

    @property
    def name(self) -> str:
        """Short identifier for the variables
        """
        if self._specs.name is not None:
            return self._specs.name
        return repr(self)

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

    def freeze(self) -> None:
        pass  # forward compatibility
