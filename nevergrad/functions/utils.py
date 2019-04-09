# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Dict, Any, Tuple, List, Optional
import numpy as np


class Transform:
    """Defines a unique random transformation (index selection, translation, and optionally rotation)
    which can be applied to a point
    """

    def __init__(self, indices: List[int], translation_factor: float = 1, rotation: bool = False) -> None:
        dim = len(indices)
        assert dim
        self.indices = np.asarray(indices)
        self.translation: np.ndarray = np.random.normal(0, 1, dim) * translation_factor
        self.rotation_matrix: Optional[np.ndarray] = None
        if rotation:
            self.rotation_matrix = np.linalg.qr(np.random.normal(0, 1, size=(dim, dim)))[0]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y: np.ndarray = x[self.indices] - self.translation
        if self.rotation_matrix is not None:
            y = self.rotation_matrix.dot(y)
        return y


class PostponedObject(abc.ABC):
    """Abstract class to inherit in order to notify the steady state benchmark executor that
    the function implements a delay. This delay will be used while benchmarking to provide the
    evaluation in a varying order.
    The main aim of this class is to make sure there is no typo in the name of the special function.

    See benchmark/execution.py for more details. This object is implemented here to avoid circular
    imports.
    """

    @abc.abstractmethod
    def get_postponing_delay(self, args: Tuple[Any, ...], kwargs: Dict[str, Any], value: float) -> float:
        raise NotImplementedError


class NoisyBenchmarkFunction(abc.ABC):
    """Mixin for use on noisy function for benchmarks.
    The noisefree_function is called at the end of benchmarks in replacement of the actual function in order
    to evaluate the final estimation. It should implement a noisefree result, or average several calls so
    that the result is as precise as possible.
    """

    @abc.abstractmethod
    def noisefree_function(self, *args: Any, **kwargs: Any) -> float:
        raise NotImplementedError
