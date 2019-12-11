# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import typing as t
import numpy as np
# importing NgDict to populate parameters (fake renaming for mypy explicit reimport)
from .core3 import Parameter


BoundValue = t.Optional[t.Union[float, int, np.int, np.float, np.ndarray]]
A = t.TypeVar("A", bound="Array")


def _check_bounds(value: np.ndarray, bounds: t.Tuple[BoundValue, BoundValue]) -> bool:
    for k, bound in enumerate(bounds):
        if bound is not None:
            if np.any((value > bound) if k else (value < bound)):
                return False
    return True


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
            recombination: t.Union[str, Parameter] = "average",
    ) -> None:
        assert isinstance(shape, tuple)
        self.shape = shape
        super().__init__(sigma=1.0, recombination=recombination)
        self._value: np.ndarray = np.zeros(shape)
        self.exponent: t.Optional[float] = None
        self.bounds: t.Tuple[np.ndarray, np.ndarray] = (None, None)
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
        if not _check_bounds(value, self.bounds):
            raise ValueError("New value does not comply with bounds")
        if self.exponent is not None and np.min(value.ravel()) <= 0:
            raise ValueError("Logirithmic values cannot be negative")
        self._value = value

    def set_bounds(self: A, a_min: BoundValue = None, a_max: BoundValue = None,
                   method: str = "clipping", full_range_sampling: bool = False) -> A:
        assert method in ["clipping"]  # , "constraint"]
        if not _check_bounds(self.value, self.bounds):
            raise ValueError("Current value is not within bounds, please update it first")
        bounds = tuple(a if isinstance(a, np.ndarray) else np.array([a], dtype=float) for a in (a_min, a_max))
        if not (a_min is None or a_max is None):
            if (bounds[0] >= bounds[1]).any():
                raise ValueError(f"Lower bounds {a_min} should be strictly smaller than upper bounds {a_max}")
        self.bounds = bounds  # type: ignore
        if method not in ["clipping", "constraint"]:
            if self.exponent is not None:
                raise ValueError(f'Cannot use method "{method}" in logarithmic mode')
            self.bounding_method = method
        if full_range_sampling and any(a is None for a in (a_min, a_max)):
            raise ValueError("Cannot use full range sampling if both bounds are not set")
        self.full_range_sampling = full_range_sampling
        # check sigma is small enough
        for name, bound in zip(("Lower", "Upper"), self.bounds):
            if bound is not None:
                std_data = self.get_std_data()
                std_bound = self._to_std_space(bound if isinstance(bound, np.ndarray) else np.array([bound]))
                min_dist = np.min(np.abs(std_data - std_bound).ravel())
                if min_dist < 2.0:
                    warnings.warn(f"{name} bound is {min_dist} sigma away from current value, "
                                  "you should aim for about 3 for better quality.")

        return self

    def set_mutation(self: A, sigma: t.Optional[t.Union[float, "Array"]] = None, exponent: t.Optional[float] = None) -> A:
        if sigma is not None:
            self.subparameters._parameters["sigma"] = sigma
        if self.exponent is None and exponent is not None:
            self.exponent = exponent
            self._value = exponent**self._value
        return self

    # pylint: disable=unused-argument
    def set_std_data(self, data: np.ndarray, deterministic: bool = True) -> None:
        assert isinstance(data, np.ndarray)
        sigma = self._get_parameter_value("sigma")
        data_reduc = (sigma * data).reshape(self._value.shape)
        self._value = data_reduc if self.exponent is None else self.exponent**data_reduc
        if self.bounding_method == "clipping":
            self._value = np.clip(self._value, self.bounds[0], self.bounds[1])

    def _internal_spawn_child(self) -> "Array":
        child = self.__class__(self.shape)
        child.subparameters._parameters = {k: v.spawn_child() if isinstance(v, Parameter) else v
                                           for k, v in self.subparameters._parameters.items()}
        child.exponent = self.exponent
        child.value = self.value
        return child

    def get_std_data(self) -> np.ndarray:
        return self._to_std_space(self._value)

    def _to_std_space(self, data: np.ndarray) -> np.ndarray:
        sigma = self._get_parameter_value("sigma")
        distribval = data if self.exponent is None else np.log(data) / np.log(self.exponent)
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

    def __init__(self) -> None:
        super().__init__(shape=(1,))

    @property  # type: ignore
    def value(self) -> float:  # type: ignore
        return self._value[0]  # type: ignore

    @value.setter
    def value(self, value: float) -> None:
        if not isinstance(value, (float, int, np.float, np.int)):
            raise TypeError(f"Received a {type(value)} in place of a scalar (float, int)")
        self._value = np.array([value], dtype=float)

    def _internal_spawn_child(self) -> "Scalar":
        child = self.__class__()
        child.subparameters._parameters = {k: v.spawn_child() if isinstance(v, Parameter) else v
                                           for k, v in self.subparameters._parameters.items()}
        child.exponent = self.exponent
        child.value = self.value
        return child


class Log(Scalar):

    def __init__(
        self,
        a_min: t.Optional[float],
        a_max: t.Optional[float],
        exponent: float = 2.0,
        init: float = 1.0,
    ) -> None:
        super().__init__()
        self.set_mutation(sigma=1.0, exponent=exponent)
        self.value = init
        self.set_bounds(a_min, a_max, method="clipping")
