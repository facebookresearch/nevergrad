# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as t
import numpy as np
# importing NgDict to populate parameters (fake renaming for mypy explicit reimport)
from .core3 import Parameter


# pylint: disable=too-many-arguments
class Array(Parameter):
    """Array variable of a given shape, on which several transforms can be applied.

    Parameters
    ----------
    sigma: float or Array
        standard deviation of a mutation
    distribution: str
        distribution of the data ("linear" or "log")
    """

    def __init__(
            self,
            shape: t.Tuple[int, ...],
            sigma: t.Union[float, "Array"] = 1.0,
            distribution: t.Union[str, Parameter] = "linear",
            recombination: t.Union[str, Parameter] = "average",
            # bounds: t.Tuple[t.Union[float, np.ndarray], t.Union[float, np.ndarray]] = (-float("inf"), float("inf")),
    ) -> None:
        assert not isinstance(shape, Parameter)
        super().__init__(shape=shape, sigma=sigma, distribution=distribution, recombination=recombination)  # , bounds=bounds)
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
    def set_std_data(self, data: np.ndarray, deterministic: bool = True) -> None:
        assert isinstance(data, np.ndarray)
        sigma = self._get_parameter_value("sigma")
        self._value = (sigma * data).reshape(self._value.shape)

    def _internal_spawn_child(self) -> "Array":
        child = super()._internal_spawn_child()
        child.value = self.value
        return child

    def get_std_data(self) -> np.ndarray:
        sigma = self._get_parameter_value("sigma")
        reduced = self._value / sigma
        return reduced.ravel()  # type: ignore

    def recombine(self, *others: "Array") -> None:
        recomb = self._get_parameter_value("recombination")
        all_p = [self] + list(others)
        if recomb == "average":
            self.set_std_data(np.mean([p.get_std_data() for p in all_p], axis=0))
        else:
            raise ValueError(f'Unknown recombination "{recomb}"')


class Scalar(Array):

    def __init__(
            self,
            sigma: t.Union[float, "Array"] = 1.0,
            distribution: t.Union[str, Parameter] = "linear",
            recombination: t.Union[str, Parameter] = "average",
    ) -> None:
        super().__init__(shape=(1,), sigma=sigma, distribution=distribution, recombination=recombination)  # , bounds=bounds)

    @property  # type: ignore
    def value(self) -> float:  # type: ignore
        return self._value[0]  # type: ignore

    @value.setter
    def value(self, value: float) -> None:
        if not isinstance(value, (float, int, np.float, np.int)):
            raise TypeError(f"Received a {type(value)} in place of a scalar (float, int)")
        self._value = np.array([value], dtype=float)

    def _internal_spawn_child(self) -> "Scalar":
        child = Scalar()
        child.subparameters._parameters = {k: v.spawn_child() if isinstance(v, Parameter) else v
                                           for k, v in self.subparameters._parameters.items()}
        child.value = self.value
        return child
