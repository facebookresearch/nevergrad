# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as t
import numpy as np
# importing NgDict to populate parameters (fake renaming for mypy explicit reimport)
from .core3 import Parameter


BoundValue = t.Optional[t.Union[float, int, np.int, np.float, np.ndarray]]
A = t.TypeVar("A", bound="Array")


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
            recombination: t.Union[str, Parameter] = "average",
    ) -> None:
        assert not isinstance(shape, Parameter)
        super().__init__(shape=shape, sigma=sigma, recombination=recombination)
        self._value: np.ndarray = np.zeros(shape)
        self.exponent: t.Optional[float] = None
        self.bounds: t.Tuple[BoundValue, BoundValue] = (None, None)
        self.bounding_method: t.Optional[str] = None
        self.full_range_sampling = False

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

    def set_bounds(self: A, a_min: BoundValue = None, a_max: BoundValue = None,
                   method: str = "clipping", full_range_sampling: bool = False) -> A:
        self.bounds = (a_min, a_max)
        if method not in ["clipping", "constraint"]:
            if self.exponent is not None:
                raise ValueError(f'Cannot use method "{method}" in logarithmic mode')
            self.bounding_method = method
        if full_range_sampling and any(a is None for a in (a_min, a_max)):
            raise ValueError("Cannot use full range sampling if both bounds are not set")
        self.full_range_sampling = full_range_sampling
        return self

    def set_mutation(self: A, sigma: t.Optional[t.Union[float, "Array"]] = None, exponent: t.Optional[float] = None) -> A:
        if sigma is not None:
            self.subparameters._parameters["sigma"] = sigma
        self.exponent = exponent
        return self

    # pylint: disable=unused-argument
    def set_std_data(self, data: np.ndarray, deterministic: bool = True) -> None:
        assert isinstance(data, np.ndarray)
        sigma = self._get_parameter_value("sigma")
        data_reduc = (sigma * data).reshape(self._value.shape)
        self._value = data_reduc if self.exponent is None else self.exponent**data_reduc

    def _internal_spawn_child(self) -> "Array":
        child = super()._internal_spawn_child()
        child.value = self.value
        return child

    def get_std_data(self) -> np.ndarray:
        sigma = self._get_parameter_value("sigma")
        distribval = self._value if self.exponent is None else np.log(self._value) / np.log(self.exponent)
        reduced = distribval / sigma
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
            recombination: t.Union[str, Parameter] = "average",
    ) -> None:
        super().__init__(shape=(1,), sigma=sigma, recombination=recombination)  # , bounds=bounds)

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
