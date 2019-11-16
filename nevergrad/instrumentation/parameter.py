from typing import Union, Tuple
import numpy as np
# importing ParametersDict to populate parameters (fake renaming for mypy explicit reimport)
# pylint: disable=unused-import,useless-import-alias
from .core3 import Parameter
from .core3 import ParametersDict as ParametersDict  # noqa


class Array(Parameter):
    """Array variable of a given shape, on which several transforms can be applied.

    Parameters
    ----------
    sigma: float or Array
        standard deviation of a mutation
    distribution: str
        distribution of the data ("linear" or "log")
    """

    def __init__(self, shape: Tuple[int, ...], sigma: Union[float, "Array"] = 1.0, distribution: Union[str, Parameter] = "linear") -> None:
        assert not isinstance(shape, Parameter)
        super().__init__(shape=shape, sigma=sigma, distribution=distribution)
        self._value: np.ndarray = np.zeros(shape)

    @property
    def value(self) -> np.ndarray:
        return self._value

    @value.setter
    def value(self, value: np.ndarray) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"Received a {type(value)} in place of a np.ndarray")
        if self._value.shape != value.shape:
            raise ValueError(f"Cannot set array of shape {self._value.shape} with value of shape {value.shape}")
        self._value = value

    # pylint: disable=unused-argument
    def with_std_data(self, data: np.ndarray, deterministic: bool = True) -> None:
        self._value = data.reshape(self.value.shape)

    def to_std_data(self) -> np.ndarray:
        return self._value.ravel()

    def spawn_child(self) -> "Array":
        child = super().spawn_child()
        child._value = self.value
        return child
