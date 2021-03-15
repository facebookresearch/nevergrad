# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This code is based on a code and ideas by Emmanuel Centeno and Antoine Moreau,
# University Clermont Auvergne, CNRS, SIGMA Clermont, Institut Pascal

from math import sqrt, tan, pi
import numpy as np
import nevergrad.common.typing as tp
import nevergrad as ng
from .. import base


def impedance_pix(x: tp.ArrayLike, dpix: float, lam: float, ep0: float, epf: float) -> float:
    """Normalized impedance Z/Z0
    ep0, epf:  epsilons in et out
    lam: lambda in nanometers
    dpix: pixel width
    """
    k0d = 2 * pi * dpix / lam
    Z = 1 / sqrt(epf)
    for n in reversed(np.sqrt(x)):  # refraction index slab
        etha = 1 / n  # bulk impedance slab
        Z = etha * (Z + 1j * etha * tan(k0d * n)) / (etha + 1j * Z * tan(k0d * n))
    R = abs((Z - 1 / sqrt(ep0)) / (Z + 1 / sqrt(ep0))) ** 2 * 100  # reflection in %
    return R


class ARCoating(base.ExperimentFunction):
    """
    Parameters
    ----------
    nbslab: int
        number of pixel layers
    d_ar: int
        depth of the structure in nm

    Notes
    -----
    - This is the minimization of reflexion, i.e. this is an anti-reflexive coating problem in normale incidence.
    - Typical parameters (nbslab, d_ar) = (10, 400) or (35, 700) for instance
     d_ar / nbslab must be at least 10
    - the function domain is R^nbslab. The values are then transformed to [epmin, epmax]^nbslab

    Credit
    ------
    This function is based on a code and ideas by Emmanuel Centeno and Antoine Moreau,
    University Clermont Auvergne, CNRS, SIGMA Clermont, Institut Pascal
    """

    def __init__(self, nbslab: int = 10, d_ar: int = 400, bounding_method: str = "bouncing") -> None:
        # Wave length range
        self.lambdas = np.arange(400, 900, 5)  # lambda values from min to max, in nm
        # AR parameters
        self.dpix = d_ar / nbslab  # width in pixels
        assert self.dpix >= 10  # dpix < 10 is physically pointless
        # input and ouptut permittivities
        self.ep0 = 1
        self.epf = 9
        self.epmin = 1
        init = (self.epmin + self.epf) / 2.0 * np.ones((nbslab,))
        sigma = (self.epf - self.ep0) / 6
        array = ng.p.Array(
            init=init,
            mutable_sigma=True,
        )
        array.set_mutation(sigma=sigma)
        array.set_bounds(self.epmin, self.epf, method=bounding_method, full_range_sampling=True)
        array.set_recombination(ng.p.mutation.Crossover(0)).set_name("")
        super().__init__(self._get_minimum_average_reflexion, array)

    def _get_minimum_average_reflexion(self, x: np.ndarray) -> float:
        x = np.array(x, copy=False).ravel()
        assert len(x) == self.dimension, f"Expected dimension {self.dimension}, got {len(x)}"
        if np.min(x) < self.epmin or np.max(x) > self.epf:  # acceptability
            return float("inf")
        value = 0.0
        for lam in self.lambdas:
            RE = impedance_pix(x, self.dpix, lam, self.ep0, self.epf)  # only normal incidence
            value = value + RE / len(self.lambdas)
        return value

    def evaluation_function(self, *recommendations: ng.p.Parameter) -> float:
        assert len(recommendations) == 1, "Should not be a pareto set for a singleobjective function"
        x = recommendations[0].value
        loss = self.function(x)
        assert isinstance(loss, float)
        base.update_leaderboard(
            f'arcoating,{self.parametrization.dimension},{self._descriptors["d_ar"]}', loss, x, verbose=True
        )
        return loss
