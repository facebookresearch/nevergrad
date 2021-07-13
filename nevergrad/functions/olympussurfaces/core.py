# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Based on https://github.com/aspuru-guzik-group/olympus

import numpy as np
from functools import partial
from nevergrad.parametrization import parameter as p
from ..base import ExperimentFunction
from olympus import surfaces  # type: ignore
from olympus import noises  # type: ignore


class OlympusSurface(ExperimentFunction):

    traditional_surfaces = {
        "Michalewicz": surfaces.Michalewicz,
        "AckleyPath": surfaces.AckleyPath,
        "Dejong": surfaces.Dejong,
        "HyperEllipsoid": surfaces.HyperEllipsoid,
        "Levy": surfaces.Levy,
        "Michalewicz": surfaces.Michalewicz,
        "Rastrigin": surfaces.Rastrigin,
        "Rosenbrock": surfaces.Rosenbrock,
        "Schwefel": surfaces.Schwefel,
        "StyblinskiTang": surfaces.StyblinskiTang,
        "Zakharov": surfaces.Zakharov,
        "DiscreteAckley": surfaces.DiscreteAckley,
        "DiscreteDoubleWell": surfaces.DiscreteDoubleWell,
        "DiscreteMichalewicz": surfaces.DiscreteMichalewicz,
        "LinearFunnel": surfaces.LinearFunnel,
        "NarrowFunnel": surfaces.NarrowFunnel,
        "GaussianMixture": surfaces.GaussianMixture,
    }

    @classmethod
    def get_surfaces_kinds(cls):
        return list(cls.traditional_surfaces)

    def __init__(
        self, kind: str, dimension: int = 10, noise_kind: str = "GaussianNoise", noise_scale: float = 1
    ) -> None:
        self.kind = kind
        self.param_dim = dimension
        self.noise_kind = noise_kind
        self.noise_scale = noise_scale
        self.surface = partial(self._simulate_surface, noise=True)
        self.surface_without_noise = partial(self._simulate_surface, noise=False)
        parametrization = p.Array(shape=(dimension,))
        parametrization.function.deterministic = False
        super().__init__(self.surface, parametrization)

    def _simulate_surface(self, x: np.ndarray, noise: bool = True) -> float:
        assert self.kind in OlympusSurface.get_surfaces_kinds()
        assert self.noise_kind in ["GaussianNoise", "UniformNoise", "GammaNoise"]

        if noise:
            noise = noises.Noise(kind=self.noise_kind, scale=self.noise_scale)
            surface = OlympusSurface.traditional_surfaces[self.kind](param_dim=self.param_dim, noise=noise)
        else:
            surface = OlympusSurface.traditional_surfaces[self.kind](param_dim=self.param_dim)
        return surface.run(x)[0][0]

    def evaluation_function(self, *recommendations) -> float:
        """Averages multiple evaluations if necessary."""
        x = recommendations[0].value
        return self.surface_without_noise(x)
