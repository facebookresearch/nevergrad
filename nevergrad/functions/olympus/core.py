# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Based on https://github.com/aspuru-guzik-group/olympus

import matplotlib.pyplot as plt
import numpy as np
from nevergrad.parametrization import parameter as p
from ..base import ExperimentFunction
import olympus  # type: ignore


class Olympus(ExperimentFunction):
    def __init__(
        self, kind: str, dimension: int = 10, noise_kind: str = "GaussianNoise", noise_scale: float = 1
    ) -> None:
        self.kind = kind
        self.param_dim = dimension
        self.noise_kind = noise_kind
        self.noise_scale = noise_scale
        self.surface_without_noise = None  # for evaluation
        parametrization = p.Array(shape=(dimension,))
        parametrization.function.deterministic = False
        super().__init__(self._simulate_traditional_surface, parametrization)

    def _simulate_traditional_surface(self, x: np.ndarray) -> float:
        assert self.kind in [
            "AckleyPath",
            "Dejong",
            "HyperEllipsoid",
            "Levy",
            "Michalewicz",
            "Rastrigin",
            "Rosenbrock",
            "Schwefel",
            "StyblinskiTang",
            "Zakharov",
        ]

        assert self.noise_kind in ["GaussianNoise", "UniformNoise", "GammaNoise"]

        traditional_surfaces = {
            "Michalewicz": olympus.surfaces.Michalewicz,
            "AckleyPath": olympus.surfaces.AckleyPath,
            "Dejong": olympus.surfaces.Dejong,
            "HyperEllipsoid": olympus.surfaces.HyperEllipsoid,
            "Levy": olympus.surfaces.Levy,
            "Michalewicz": olympus.surfaces.Michalewicz,
            "Rastrigin": olympus.surfaces.Rastrigin,
            "Rosenbrock": olympus.surfaces.Rosenbrock,
            "Schwefel": olympus.surfaces.Schwefel,
            "StyblinskiTang": olympus.surfaces.StyblinskiTang,
            "Zakharov": olympus.surfaces.Zakharov,
        }
        noise = olympus.noises.Noise(kind=self.noise_kind, scale=self.noise_scale)
        surface = traditional_surfaces[self.kind](param_dim=self.param_dim, noise=noise)
        self.surface_without_noise = traditional_surfaces[self.kind](param_dim=self.param_dim)
        return surface.run(x)[0][0]

    def evaluation_function(self, *recommendations) -> float:
        """Averages multiple evaluations if necessary."""
        x = recommendations[0].value
        losses = [self.surface_without_noise.run(x)[0][0] for _ in range(42)]
        return sum(losses) / len(losses)
