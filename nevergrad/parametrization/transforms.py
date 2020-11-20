# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import uuid
import itertools
import numpy as np
from scipy import stats
import nevergrad.common.typing as tp


class Transform:
    """Base class for transforms implementing a forward and a backward (inverse)
    method.
    This provide a default representation, and a short representation should be implemented
    for each transform.
    """

    def __init__(self) -> None:
        self.name = uuid.uuid4().hex  # a name for easy identification. This random uuid should be overriden

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reverted(self) -> 'Transform':
        return Reverted(self)

    def __repr__(self) -> str:
        args = ", ".join(f"{x}={y}" for x, y in sorted(self.__dict__.items()) if not x.startswith("_"))
        return f"{self.__class__.__name__}({args})"


class Reverted(Transform):
    """Inverse of a transform.

    Parameters
    ----------
    transform: Transform
    """

    def __init__(self, transform: Transform) -> None:
        super().__init__()
        self.transform = transform
        self.name = f"Rv({self.transform.name})"

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.transform.backward(x)

    def backward(self, y: np.ndarray) -> np.ndarray:
        return self.transform.forward(y)


class Affine(Transform):
    """Affine transform a * x + b

    Parameters
    ----------
    a: float
    b: float
    """

    def __init__(self, a: float, b: float) -> None:
        super().__init__()
        if not a:
            raise ValueError('"a" parameter should be non-zero to prevent information loss.')
        self.a = a
        self.b = b
        self.name = f"Af({self.a},{self.b})"

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.a * x + self.b  # type: ignore

    def backward(self, y: np.ndarray) -> np.ndarray:
        return (y - self.b) / self.a  # type: ignore


class Exponentiate(Transform):
    """Exponentiation transform base ** (coeff * x)
    This can for instance be used for to get a logarithmicly distruted values 10**(-[1, 2, 3]).

    Parameters
    ----------
    base: float
    coeff: float
    """

    def __init__(self, base: float = 10., coeff: float = 1.) -> None:
        super().__init__()
        self.base = base
        self.coeff = coeff
        self.name = f"Ex({self.base},{self.coeff})"

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.base ** (float(self.coeff) * x)  # type: ignore

    def backward(self, y: np.ndarray) -> np.ndarray:
        return np.log(y) / (float(self.coeff) * np.log(self.base))  # type: ignore


BoundType = tp.Optional[tp.Union[tp.ArrayLike, float]]


def _f(x: BoundType) -> BoundType:
    """Format for prints:
    array with one scalars are converted to floats
    """
    if isinstance(x, (np.ndarray, list, tuple)):
        x = np.array(x, copy=False)
        if x.shape == (1,):
            x = float(x[0])
    if isinstance(x, float) and x.is_integer():
        x = int(x)
    return x


class BoundTransform(Transform):  # pylint: disable=abstract-method

    def __init__(
        self,
        a_min: BoundType = None,
        a_max: BoundType = None
    ) -> None:
        super().__init__()
        self.a_min: tp.Optional[np.ndarray] = None
        self.a_max: tp.Optional[np.ndarray] = None
        for name, value in [("a_min", a_min), ("a_max", a_max)]:
            if value is not None:
                isarray = isinstance(value, (tuple, list, np.ndarray))
                setattr(self, name, np.array(value, copy=False) if isarray else np.array([value]))
        if not (self.a_min is None or self.a_max is None):
            if (self.a_min >= self.a_max).any():
                raise ValueError(f"Lower bounds {a_min} should be strictly smaller than upper bounds {a_max}")
        if self.a_min is None and self.a_max is None:
            raise ValueError("At least one bound must be specified")
        self.shape: tp.Tuple[int, ...] = self.a_min.shape if self.a_min is not None else self.a_max.shape

    def _check_shape(self, x: np.ndarray) -> None:
        for dims in itertools.zip_longest(x.shape, self.shape, fillvalue=1):
            if dims[0] != dims[1] and not any(x == 1 for x in dims):  # same or broadcastable
                raise ValueError(f"Shapes do not match: {self.shape} and {x.shape}")


class TanhBound(BoundTransform):
    """Bounds all real values into [a_min, a_max] using a tanh transform.
    Beware, tanh goes very fast to its limits.

    Parameters
    ----------
    a_min: float
    a_max: float
    """

    def __init__(
        self,
        a_min: tp.Union[tp.ArrayLike, float],
        a_max: tp.Union[tp.ArrayLike, float]
    ) -> None:
        super().__init__(a_min=a_min, a_max=a_max)
        if self.a_min is None or self.a_max is None:
            raise ValueError("Both bounds must be specified")
        self._b = .5 * (self.a_max + self.a_min)
        self._a = .5 * (self.a_max - self.a_min)
        self.name = f"Th({_f(a_min)},{_f(a_max)})"

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._check_shape(x)
        return self._b + self._a * np.tanh(x)  # type: ignore

    def backward(self, y: np.ndarray) -> np.ndarray:
        self._check_shape(y)
        if (y > self.a_max).any() or (y < self.a_min).any():
            raise ValueError(f"Only data between {self.a_min} and {self.a_max} "
                             "can be transformed back (bounds lead to infinity).")
        return np.arctanh((y - self._b) / self._a)  # type: ignore


class Clipping(BoundTransform):
    """Bounds all real values into [a_min, a_max] using clipping (not bijective).

    Parameters
    ----------
    a_min: float or None
        lower bound
    a_max: float or None
        upper bound
    bounce: bool
        bounce (once) on borders instead of just clipping
    """

    def __init__(
        self,
        a_min: BoundType = None,
        a_max: BoundType = None,
        bounce: bool = False,
    ) -> None:
        super().__init__(a_min=a_min, a_max=a_max)
        self._bounce = bounce
        b = ",b" if bounce else ""
        self.name = f"Cl({_f(a_min)},{_f(a_max)}{b})"

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._check_shape(x)
        out = np.clip(x, self.a_min, self.a_max)
        if self._bounce:
            out = np.clip(2 * out - x, self.a_min, self.a_max)
        return out  # type: ignore

    def backward(self, y: np.ndarray) -> np.ndarray:
        self._check_shape(y)
        if (self.a_max is not None and (y > self.a_max).any()) or (self.a_min is not None and (y < self.a_min).any()):
            raise ValueError(f"Only data between {self.a_min} and {self.a_max} "
                             "can be transformed back.")
        return y


class ArctanBound(BoundTransform):
    """Bounds all real values into [a_min, a_max] using an arctan transform.
    This is a much softer approach compared to tanh.

    Parameters
    ----------
    a_min: float
    a_max: float
    """

    def __init__(
        self,
        a_min: tp.Union[tp.ArrayLike, float],
        a_max: tp.Union[tp.ArrayLike, float]
    ) -> None:
        super().__init__(a_min=a_min, a_max=a_max)
        if self.a_min is None or self.a_max is None:
            raise ValueError("Both bounds must be specified")
        self._b = .5 * (self.a_max + self.a_min)
        self._a = (self.a_max - self.a_min) / np.pi
        self.name = f"At({_f(a_min)},{_f(a_max)})"

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._check_shape(x)
        return self._b + self._a * np.arctan(x)

    def backward(self, y: np.ndarray) -> np.ndarray:
        self._check_shape(y)
        if (y > self.a_max).any() or (y < self.a_min).any():
            raise ValueError(f"Only data between {self.a_min} and {self.a_max} can be transformed back.")
        return np.tan((y - self._b) / self._a)


class CumulativeDensity(BoundTransform):
    """Bounds all real values into [0, 1] using a gaussian cumulative density function (cdf)
    Beware, cdf goes very fast to its limits.
    """

    def __init__(
        self,
        lower: float = 0.0,
        upper: float = 1.0,
        eps: float = 1e-9
    ) -> None:
        super().__init__(a_min=lower, a_max=upper)
        self._b = lower
        self._a = upper - lower
        self._eps = eps
        self.name = f"Cd({_f(lower)},{_f(upper)})"

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._a * stats.norm.cdf(x) + self._b

    def backward(self, y: np.ndarray) -> np.ndarray:
        if (y > self.a_max).any() or (y < self.a_min).any():
            raise ValueError(f"Only data between {self.a_min} and {self.a_max} can be transformed back.\nGot: {y}")
        y = np.clip((y - self._b) / self._a, self._eps, 1 - self._eps)
        return stats.norm.ppf(y)


class Fourrier(Transform):

    def __init__(
        self,
        axes: tp.Union[int, tp.Sequence[int]] = 0
    ) -> None:
        super().__init__()
        self.axes: tp.Tuple[int, ...] = (axes,) if isinstance(axes, int) else tuple(axes)  # type: ignore
        self.name = f"F({axes})"

    def forward(self, x: np.ndarray) -> np.ndarray:
        if any(x.shape[a] % 2 for a in self.axes):
            raise ValueError(f"Only even shapes are allowed for Fourrier transform, got {x.shape}")
        return np.fft.rfftn(x, axes=self.axes, norm="ortho")  # type: ignore

    def backward(self, y: np.ndarray) -> np.ndarray:
        return np.fft.irfftn(y, axes=self.axes, norm="ortho")  # type: ignore
