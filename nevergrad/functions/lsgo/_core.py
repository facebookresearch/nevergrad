# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This is based on the specifications of:
# X. Li, K. Tang, M. Omidvar, Z. Yang and K. Qin,
# "Benchmark Functions for the CEC’2013 Special Session and Competition on Large Scale Global Optimization",
# Technical Report, Evolutionary Computation and Machine Learning Group, RMIT University, Australia, 2013.
# and the corresponding source code.
from typing import List, Callable, Optional
from math import exp, sqrt
import numpy as np


def sphere(x: np.ndarray) -> float:
    assert x.ndim == 1
    return float(x.dot(x))


class Elliptic:
    def __init__(self, dimension: int) -> None:
        self.weights = 10 ** np.linspace(0, 6, dimension)  # precompute for speed

    def __call__(self, x: np.ndarray) -> float:
        return float(self.weights.dot(x**2))


def elliptic(x: np.ndarray) -> float:
    return Elliptic(x.size)(x)


def rastrigin(x: np.ndarray) -> float:
    sum_cos = float(np.sum(np.cos(2 * np.pi * x)))
    return 10.0 * (x.size - sum_cos) + sphere(x)


def ackley(x: np.ndarray) -> float:
    dim = x.size
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    return -20.0 * exp(-0.2 * sqrt(sphere(x) / dim)) - exp(sum_cos / dim) + 20 + exp(1)


def schwefel_1_2(x: np.ndarray) -> float:
    cx = np.cumsum(x)
    return sphere(cx)


def rosenbrock(x: np.ndarray) -> float:
    x_m_1 = x[:-1] - 1
    x_diff = x[:-1] ** 2 - x[1:]
    return float(100 * x_diff.dot(x_diff) + x_m_1.dot(x_m_1))


# %% Transformations


def irregularity(x: np.ndarray) -> np.ndarray:  # Tosz
    xhat = np.abs(x)
    xhat[x == 0.0] = 1.0  # get rid of 0 special case
    xhat = np.log(xhat)
    signx = np.sign(x, dtype=float)
    c1 = 0.5 * (10.0 + 5.5) + 0.5 * (10 - 5.5) * signx  # 5.5 if negative else 10
    c2 = 0.5 * (3.1 + 7.9) + 0.5 * (7.9 - 3.1) * signx  # 3.1 if negative else 7.9
    # the following seems slightly faster than positive then negative
    output: np.ndarray = signx * np.exp(xhat + 0.049 * (np.sin(c1 * xhat) + np.sin(c2 * xhat)))
    return output


class Asymmetry:  # Tasy
    def __init__(self, beta: float = 0.2) -> None:
        self.beta = beta
        self._weights: Optional[np.ndarray] = None

    def _get_weights(self, dimension: int) -> np.ndarray:
        if self._weights is None:  # caching for speed
            self._weights = np.linspace(0, self.beta, dimension)
        return self._weights

    def __call__(self, x: np.ndarray) -> np.ndarray:
        exponents = 1 + self._get_weights(x.size) * np.sqrt(np.maximum(0, x))
        return x**exponents  # type: ignore


class Illconditionning:  # Lambda matrix
    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = alpha
        self._weights: Optional[np.ndarray] = None

    def _get_weights(self, dimension: int) -> np.ndarray:
        if self._weights is None:  # caching for speed
            self._weights = self.alpha ** np.linspace(0, 0.5, num=dimension)
        return self._weights

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self._get_weights(x.size) * x  # type: ignore


class Translation:
    def __init__(self, translation: np.ndarray) -> None:
        self.translation = translation

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x - self.translation  # type: ignore


class Indexing:
    def __init__(self, indices: np.ndarray, indim: int) -> None:
        self.indices = indices
        self.outdim = indices.size
        self.indim = indim

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not x.size == self.indim:
            raise ValueError(f"Got size {x.size} but expected {self.indim}")
        return x[self.indices]  # type: ignore

    @classmethod
    def from_split(cls, permutation: np.ndarray, dimensions: List[int], overlap: int = 0) -> List["Indexing"]:
        indexing = split(permutation, dimensions, overlap)
        return [cls(inds, permutation.size) for inds in indexing]


class Rotation:
    def __init__(self, rotation: np.ndarray) -> None:
        shape = rotation.shape
        assert len(shape) == 2 and shape[0] == shape[1]
        self.rotation = rotation

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.rotation.dot(x)  # type: ignore

    @classmethod
    def from_random(cls, dimension: int, random_state: Optional[np.random.RandomState] = None) -> "Rotation":
        if random_state is None:
            random_state = np.random  # type: ignore
        return cls(np.linalg.qr(random_state.normal(0, 1, size=(dimension, dimension)))[0])  # type: ignore


def split(permutation: np.ndarray, dimensions: List[int], overlap: int = 0) -> List[np.ndarray]:
    expected_size = sum(dimensions) - (len(dimensions) - 1) * overlap
    assert len(permutation) == len(set(permutation))
    assert min(permutation) == 0
    assert max(permutation) == len(permutation) - 1
    if permutation.size != expected_size:
        raise ValueError(
            f"Permutation should have size {expected_size} for dimensions {dimensions} with overlap {overlap}"
        )
    pattern: List[np.ndarray] = []
    start_ind = 0
    for length in dimensions:
        pattern.append(permutation[start_ind : start_ind + length])
        start_ind += length - overlap
    assert start_ind == permutation.size - overlap
    return pattern


def apply_transforms(x: np.ndarray, transforms: List[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    y = transforms[0](x)
    for transf in transforms[1:]:
        y = transf(y)
    return y
