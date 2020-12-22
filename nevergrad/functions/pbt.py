# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import numpy as np
from nevergrad.parametrization import parameter as p
from .base import ExperimentFunction
from . import corefuncs


class PBT(ExperimentFunction):
    """Population-Based Training, also known as Lamarckism or Meta-Optimization."""

    def __init__(
        self,
        names: tp.Tuple[str, ...] = ("sphere", "cigar", "ellipsoid"),
        dimensions: tp.Tuple[int, ...] = (7, 7, 7),
        num_workers: int = 10,
    ):
        for name in names:
            if name not in corefuncs.registry:
                available = ", ".join(sorted(corefuncs.registry))
                raise ValueError(
                    f'Unknown core function "{name}" in PBT. Available names are:\n-----\n{available}'
                )
        self._funcs = [corefuncs.registry[name] for name in names]
        self._optima = [np.random.normal(size=d) for d in dimensions]
        assert len(names) == len(dimensions)
        self._hyperparameter_dimension = len(names)
        self._dimensions = dimensions
        self._total_dimension = sum(dimensions)
        parametrization = p.Array(shape=(self._hyperparameter_dimension,)).set_name("")
        # Population of checkpoints (that are optimized by the underlying optimization method)
        # and parameters (that we do optimize).
        self._population_checkpoints: tp.List[np.ndarray] = [np.zeros(self._total_dimension)] * num_workers
        self._population_parameters: tp.List[np.ndarray] = [
            np.zeros(self._hyperparameter_dimension)
        ] * num_workers
        self._population_fitness: tp.List[float] = [float("inf")] * num_workers
        super().__init__(self._func, parametrization)

    # The 3 methods below are function-specific.
    def unflatten(self, x):
        y = []
        current_idx = 0
        for i in range(self._hyperparameter_dimension):
            y += [x[current_idx : (current_idx + self._dimensions[i])]]
            current_idx += self._dimensions[i]
        assert current_idx == len(x) == self._total_dimension
        return y

    def value(self, x):
        y = self.unflatten(x)
        return sum(f(xi - o) for f, xi, o in zip(self._funcs, y, self._optima))

    def evolve(self, x: np.ndarray, p: np.ndarray):
        assert len(p) == self._hyperparameter_dimension

        # We define a gradient operator.
        def gradient(f, x):
            epsilon = 1e-15
            # We compute a gradient by finite differences.
            g = np.zeros(len(x))
            value_minus = f(x)
            assert type(value_minus) == type(1.5), str(type(value_minus))
            for i in range(len(x)):
                e = np.zeros(len(x))
                e[i] = epsilon
                value_plus = f(x + e)
                assert type(value_plus) == type(1.5), str(type(value_plus))
                g[i] = (value_plus - value_minus) / epsilon
            return g

        y = self.unflatten(x)
        assert len(y) == self._hyperparameter_dimension

        # There are self._hyperparameter_dimension learning rates.
        # Each of them is responsible of one gradient descent.
        for j in range(self._hyperparameter_dimension):
            assert len(y[j]) == self._dimensions[j]
            assert type(self._funcs[j](y[j])) == type(1.0), str(type(self._funcs[j](y[j])))
            y[j] -= np.exp(p[j]) * (
                gradient(self._funcs[j], y[j] - self._optima[j]) + np.random.normal(self._dimensions[j])
            )
        current_idx = 0
        for j in range(self._hyperparameter_dimension):
            x[current_idx : (current_idx) + self._dimensions[j]] = y[j]
            current_idx += self._dimensions[j]
        assert current_idx == self._total_dimension

    def _func(self, x: np.ndarray):
        assert len(x) == self._hyperparameter_dimension

        # First, let us find the checkpoint that we want to use.
        if np.random.uniform() > 0.666:
            distances = self._population_fitness
        else:
            if np.random.uniform() > 0.5:
                distances = [np.linalg.norm(i - x, 0) for i in self._population_parameters]
            else:
                distances = [np.linalg.norm(i - x, 1) for i in self._population_parameters]
        _, source_idx = min((val, idx) for (idx, val) in enumerate(distances))

        # Let us copy the checkpoint to a target.
        idx = np.random.choice(range(len(self._population_fitness)))
        if idx != source_idx:
            self._population_checkpoints[idx] = self._population_checkpoints[source_idx].copy()
            self._population_fitness[idx] = self._population_fitness[source_idx]
        self._population_parameters[idx] = x

        # Here the case-specific learning and evaluation.
        self.evolve(self._population_checkpoints[idx], self._population_parameters[idx])
        self._population_fitness[idx] = self.value(self._population_checkpoints[idx])

        return self._population_fitness[idx]

    @classmethod
    def itercases(cls) -> tp.Iterator["PBT"]:
        options = dict(
            names=[["Sphere", "Cigar", "Ellipsoid", "Cigar"], ["Hm", "Rastrigin", "Sphere", "Cigar"]],
            dimensions=[[2, 2, 2, 2], [5, 5, 5, 5], [77, 77, 77, 77]],
        )
        keys = sorted(options)
        select = itertools.product(*(options[k] for k in keys))  # type: ignore
        cases = (dict(zip(keys, s)) for s in select)
        return (cls(**c) for c in cases)
