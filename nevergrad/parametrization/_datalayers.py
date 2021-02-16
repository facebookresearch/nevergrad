# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import functools
import numpy as np
import nevergrad.common.typing as tp
from nevergrad.common import errors
from . import _layering
from .data import Data
from .core import Parameter
from . import transforms as trans
from . import utils


D = tp.TypeVar("D", bound=Data)
BL = tp.TypeVar("BL", bound="BoundLayer")


class BoundLayer(_layering.Layered):

    _LAYER_LEVEL = _layering.Level.OPERATION

    def __init__(
        self,
        lower: tp.BoundValue = None,
        upper: tp.BoundValue = None,
        uniform_sampling: tp.Optional[bool] = None,
    ) -> None:
        """Bounds all real values into [lower, upper]

        Parameters
        ----------
        lower: float or None
            minimum value
        upper: float or None
            maximum value
        method: str
            One of the following choices:
        uniform_sampling: Optional bool
            Changes the default behavior of the "sample" method (aka creating a child and mutating it from the current instance)
            or the sampling optimizers, to creating a child with a value sampled uniformly (or log-uniformly) within
            the while range of the bounds. The "sample" method is used by some algorithms to create an initial population.
            This is activated by default if both bounds are provided.
        """  # TODO improve description of methods
        super().__init__()
        self.bounds = tuple(
            a if isinstance(a, np.ndarray) or a is None else np.array([a], dtype=float)
            for a in (lower, upper)
        )
        both_bounds = all(b is not None for b in self.bounds)
        self.uniform_sampling = uniform_sampling
        if uniform_sampling is None:
            self.uniform_sampling = both_bounds
        if self.uniform_sampling and not both_bounds:
            raise errors.NevergradValueError("Cannot use full range sampling if both bounds are not set")
        if not (lower is None or upper is None):
            if (self.bounds[0] >= self.bounds[1]).any():  # type: ignore
                raise errors.NevergradValueError(
                    f"Lower bounds {lower} should be strictly smaller than upper bounds {upper}"
                )

    def __call__(self, data: D, inplace: bool = False) -> D:
        """Creates a new Data instance with bounds"""
        new = data if inplace else data.copy()
        # if not utils.BoundChecker(*self.bounds)(new.value):
        #     raise errors.NevergradValueError("Current value is not within bounds, please update it first")
        value = new.value
        new.add_layer(self.copy())
        try:
            new.value = value
        except ValueError as e:
            raise errors.NevergradValueError(
                "Current value is not within bounds, please update it first"
            ) from e
        # TODO warn if sigma is too large for range
        # if all(x is not None for x in self.bounds):
        #     std_bounds = tuple(self._to_reduced_space(b) for b in self.bounds)  # type: ignore
        #     min_dist = np.min(np.abs(std_bounds[0] - std_bounds[1]).ravel())
        #     if min_dist < 3.0:
        #         warnings.warn(
        #             f"Bounds are {min_dist} sigma away from each other at the closest, "
        #             "you should aim for at least 3 for better quality."
        #         )
        return new

    def _layered_sample(self) -> "Data":
        if not self.uniform_sampling:
            return super()._layered_sample()  # type: ignore
        root = self._layers[0]
        if not isinstance(root, Data):
            raise errors.NevergradTypeError(f"BoundLayer {self} on a non-Data root {root}")
        child = root.spawn_child()
        shape = super()._layered_get_value().shape
        bounds = tuple(b * np.ones(shape) for b in self.bounds)
        new_val = root.random_state.uniform(*bounds)
        # send new val to the layer under this one for the child
        child._deeper_layers()[-1]._layered_set_value(new_val)
        return child

    def _check(self, value: np.ndarray) -> None:
        if not utils.BoundChecker(*self.bounds)(value):
            raise errors.NevergradValueError("New value does not comply with bounds")


class Modulo(BoundLayer):
    """Applies a modulo operation on the array
    CAUTION: WIP
    """

    def __init__(self, module: tp.Any) -> None:
        super().__init__(lower=0, upper=module)
        if not isinstance(module, (np.ndarray, np.float, np.int, float, int)):
            raise TypeError(f"Unsupported type {type(module)} for module")
        self._module = module

    def _layered_get_value(self) -> np.ndarray:
        return super()._layered_get_value() % self._module  # type: ignore

    def _layered_set_value(self, value: np.ndarray) -> None:
        self._check(value)
        current = super()._layered_get_value()
        super()._layered_set_value(current - (current % self._module) + value)


class Operation(_layering.Layered):

    _LAYER_LEVEL = _layering.Level.OPERATION

    def __init__(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        super().__init__()
        if any(isinstance(x, Parameter) for x in args + tuple(kwargs.values())):
            raise errors.NevergradTypeError("Operation with Parameter instances are not supported")


class Exponent(Operation):
    """Applies an array as exponent of a float"""

    def __init__(self, base: float) -> None:
        super().__init__(base)
        if base <= 0:
            raise errors.NevergradValueError("Exponent must be strictly positive")
        self._base = base
        self.set_name(f"exp={base}")

    def _layered_get_value(self) -> np.ndarray:
        return self._base ** super()._layered_get_value()  # type: ignore

    def _layered_set_value(self, value: np.ndarray) -> None:
        super()._layered_set_value(np.log(value) / np.log(self._base))


class Power(Operation):
    """Applies a float as exponent of a Data parameter"""

    def __init__(self, power: float) -> None:
        super().__init__(power)
        self._power = power

    def _layered_get_value(self) -> np.ndarray:
        return super()._layered_get_value() ** self._power  # type: ignore

    def _layered_set_value(self, value: np.ndarray) -> None:
        super()._layered_set_value(value ** (1.0 / self._power))


class Add(Operation):
    """Applies an array as exponent of a floar"""

    def __init__(self, offset: tp.Any) -> None:
        super().__init__(offset)
        self._offset = offset

    def _layered_get_value(self) -> np.ndarray:
        return self._offset + super()._layered_get_value()  # type: ignore

    def _layered_set_value(self, value: np.ndarray) -> None:
        super()._layered_set_value(value - self._offset)


class Multiply(Operation):
    """Applies an array as exponent of a floar"""

    def __init__(self, value: tp.Any) -> None:
        super().__init__(value)
        self._value = value

    def _layered_get_value(self) -> np.ndarray:
        return self._value * super()._layered_get_value()  # type: ignore

    def _layered_set_value(self, value: np.ndarray) -> None:
        super()._layered_set_value(value / self._value)


class Bound(BoundLayer):
    def __init__(
        self,
        lower: tp.BoundValue = None,
        upper: tp.BoundValue = None,
        method: str = "bouncing",
        uniform_sampling: tp.Optional[bool] = None,
    ) -> None:
        """Bounds all real values into [lower, upper] using a provided method

        See Parameter.set_bounds
        """
        super().__init__(lower=lower, upper=upper, uniform_sampling=uniform_sampling)
        # update instance
        transforms = dict(
            clipping=trans.Clipping,
            arctan=trans.ArctanBound,
            tanh=trans.TanhBound,
            gaussian=trans.CumulativeDensity,
        )
        transforms["bouncing"] = functools.partial(trans.Clipping, bounce=True)  # type: ignore
        if method not in transforms:
            raise errors.NevergradValueError(
                f"Unknown method {method}, available are: {transforms.keys()}\nSee docstring for more help."
            )
        self._transform = transforms[method](*self.bounds)
        self.set_name(self._transform.name)

    def _layered_get_value(self) -> np.ndarray:
        return self._transform.forward(super()._layered_get_value())  # type: ignore

    def _layered_set_value(self, value: np.ndarray) -> None:
        super()._layered_set_value(self._transform.backward(value))
