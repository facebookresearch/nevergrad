# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Based on https://github.com/aspuru-guzik-group/olympus

import matplotlib.pyplot as plt
import numpy as np
from nevergrad.parametrization import parameter as p
from ..base import ExperimentFunction
import olympus


class Olympus(ExperimentFunction):
    def __init__(self, kind : str, dimension: int = 10) -> None:
        self.kind = kind
        self.param_dim = dimension
        super().__init__(self._simulate_traditional_surface, p.Array(shape=(dimension,)))
    
    def _simulate_traditional_surface(self, x: np.ndarray) -> float:
        assert self.kind in ["AckleyPath", "Dejong", "HyperEllipsoid",
                        "Levy", "Michalewicz", "Rastrigin", "Rosenbrock", 
                        "Schwefel", "StyblinskiTang", "Zakharov"]

        traditional_surfaces = {
            "Michalewicz":olympus.surfaces.Michalewicz,
            "AckleyPath":olympus.surfaces.AckleyPath,
            "Dejong":olympus.surfaces.Dejong,
            "HyperEllipsoid":olympus.surfaces.HyperEllipsoid,
            "Levy":olympus.surfaces.Levy,
            "Michalewicz":olympus.surfaces.Michalewicz,
            "Rastrigin":olympus.surfaces.Rastrigin,
            "Rosenbrock":olympus.surfaces.Rosenbrock,
            "Schwefel":olympus.surfaces.Schwefel,
            "StyblinskiTang":olympus.surfaces.StyblinskiTang,
            "Zakharov":olympus.surfaces.Zakharov,
        }
        
        surface = traditional_surfaces[self.kind](param_dim=self.param_dim)
        return surface.run(x)[0][0]