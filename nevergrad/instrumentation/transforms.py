import numpy as np


class Transform:

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def revert(self) -> 'Transform':
        return Revert(self)

    def _short_repr(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        args = ", ".join(f"{x}={y}" for x, y in sorted(self.__dict__.items()) if not x.startswith("_"))
        return f"{self.__class__.__name__}({args})"

    def __format__(self, format_spec: str) -> str:
        if format_spec == "short":
            return self._short_repr()
        return repr(self)


class Revert(Transform):

    def __init__(self, transform: Transform) -> None:
        self.transform = transform

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.transform.backward(x)

    def backward(self, y: np.ndarray) -> np.ndarray:
        return self.transform.forward(y)

    def _short_repr(self) -> str:
        return f'Rv({self.transform})'


class Affine(Transform):

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

    def __init__(self, min_val: float, max_val: float) -> None:
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

    def __init__(self, min_val: float, max_val: float) -> None:
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
