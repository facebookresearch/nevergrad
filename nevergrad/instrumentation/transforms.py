# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from scipy import stats


class Transform:
    """Base class for transforms implementing a forward and a backward (inverse)
    method.
    This provide a default representation, and a short representation should be implemented
    for each transform.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reverted(self) -> 'Transform':
        return Reverted(self)

    def _short_repr(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        args = ", ".join(f"{x}={y}" for x, y in sorted(self.__dict__.items()) if not x.startswith("_"))
        return f"{self.__class__.__name__}({args})"

    def __format__(self, format_spec: str) -> str:
        if format_spec == "short":
            return self._short_repr()
        return repr(self)


class Reverted(Transform):
    """Inverse of a transform.

    Parameters
    ----------
    transform: Transform
    """

    def __init__(self, transform: Transform) -> None:
        self.transform = transform

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.transform.backward(x)

    def backward(self, y: np.ndarray) -> np.ndarray:
        return self.transform.forward(y)

    def _short_repr(self) -> str:
        return f'Rv({self.transform:short})'


class Affine(Transform):
    """Affine transform a * x + b

    Parameters
    ----------
    a: float
    b: float
    """

    def __init__(self, a: float, b: float) -> None:
        self.a = a
        self.b = b

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.a * x + self.b  # type: ignore

    def backward(self, y: np.ndarray) -> np.ndarray:
        return (y - self.b) / self.a  # type: ignore

    def _short_repr(self) -> str:
        return f"Af({self.a},{self.b})"


class Exponentiate(Transform):
    """Exponentiation transform base ** (coeff * x)
    This can for instance be used for to get a logarithmicly distruted values 10**(-[1, 2, 3]).

    Parameters
    ----------
    base: float
    coeff: float
    """

    def __init__(self, base: float = 10., coeff: float = 1.) -> None:
        self.base = base
        self.coeff = coeff

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.base ** (float(self.coeff) * x)  # type: ignore

    def backward(self, y: np.ndarray) -> np.ndarray:
        return np.log(y) / (float(self.coeff) * np.log(self.base))  # type: ignore

    def _short_repr(self) -> str:
        return f"Ex({self.base},{self.coeff})"


class TanhBound(Transform):
    """Bounds all real values into [min_val, max_val] using a tanh transform.
    Beware, tanh goes very fast to its limits.

    Parameters
    ----------
    min_val: float
    max_val: float
    """

    def __init__(self, min_val: float, max_val: float) -> None:
        assert min_val < max_val
        self.min_val = min_val
        self.max_val = max_val
        self._b = .5 * (self.max_val + self.min_val)
        self._a = .5 * (self.max_val - self.min_val)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._b + self._a * np.tanh(x)  # type: ignore

    def backward(self, y: np.ndarray) -> np.ndarray:
        return np.arctanh((y - self._b) / self._a)  # type: ignore

    def _short_repr(self) -> str:
        return f"Th({self.min_val},{self.max_val})"


class ArctanBound(Transform):
    """Bounds all real values into [min_val, max_val] using an arctan transform.
    This is a much softer approach compared to tanh.

    Parameters
    ----------
    min_val: float
    max_val: float
    """

    def __init__(self, min_val: float, max_val: float) -> None:
        assert min_val < max_val
        self.min_val = min_val
        self.max_val = max_val
        self._b = .5 * (self.max_val + self.min_val)
        self._a = (self.max_val - self.min_val) / np.pi

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._b + self._a * np.arctan(x)  # type: ignore

    def backward(self, y: np.ndarray) -> np.ndarray:
        return np.tan((y - self._b) / self._a)  # type: ignore

    def _short_repr(self) -> str:
        return f"At({self.min_val},{self.max_val})"


class CumulativeDensity(Transform):
    """Bounds all real values into [0, 1] using a gaussian cumulative density function (cdf)
    Beware, cdf goes very fast to its limits.
    """

    def forward(self, x: np.ndarray) -> np.ndarray:
        return stats.norm.cdf(x)  # type: ignore

    def backward(self, y: np.ndarray) -> np.ndarray:
        return stats.norm.ppf(y)  # type: ignore

    def _short_repr(self) -> str:
        return f"Cd()"
