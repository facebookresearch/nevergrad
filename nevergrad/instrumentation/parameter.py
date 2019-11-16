import numpy as np
# importing ParametersDict to populate parameters (fake renaming for mypy explicit reimport)
# pylint: disable=unused-import,useless-import-alias
from .core3 import Parameter
from .core3 import ParametersDict as ParametersDict  # noqa


class Array(Parameter):
    """Array variable of a given shape, on which several transforms can be applied.
    """

    def __init__(self, *dims: int) -> None:
        super().__init__()
        self._value: np.ndarray = np.zeros(dims)

    @property
    def value(self) -> np.ndarray:
        return self._value

    def with_value(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Received a {type(value)} in place of a np.ndarray")
        if self._value.shape != value.shape:
            raise ValueError(f"Cannot set array of shape {self._value.shape} with value of shape {value.shape}")
        self._value = value

    def with_std_data(self, data: np.ndarray, deterministic: bool = True) -> None:
        self._value = data.reshape(self.value.shape)

    def to_std_data(self) -> np.ndarray:
        return self._value.ravel()

    def spawn_child(self) -> "Array":
        child = Array(*self.value.shape)
        child._value = self.value
        child.parents_uids.append(self.uid)
        return child
