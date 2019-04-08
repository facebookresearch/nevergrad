# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This code is based on a code and ideas by Emmanuel Centeno and Antoine Moreau,
# University Clermont Auvergne, CNRS, SIGMA Clermont, Institut Pascal

from math import sqrt, tan, pi
import numpy as np
from nevergrad.common.typetools import ArrayLike
from ... import instrumentation as inst


def impedance_pix(x: ArrayLike, dpix: float, lam: float, ep0: float, epf: float) -> float:
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
    R = abs((Z - 1 / sqrt(ep0)) / (Z + 1 / sqrt(ep0)))**2 * 100  # reflection in %
    return R


class ARCoatingVariable(inst.var.utils.Variable[np.ndarray]):

    def __init__(self, dimension: int, epmin: float, epf: float) -> None:
        self._dimension = dimension
        self.epf = epf
        self.epmin = epmin

    @property
    def dimension(self) -> int:
        return self._dimension

    def data_to_argument(self, data: ArrayLike, deterministic: bool = True) -> np.ndarray:  # pylint: disable=unused-argument
        return (self.epf - self.epmin) * .5 * (1 + np.tanh(data)) + self.epmin  # type: ignore

    def _short_repr(self) -> str:
        return "Photonics"


class ARCoating(inst.InstrumentedFunction):
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

    def __init__(self, nbslab: int = 10, d_ar: int = 400) -> None:
        # Wave length range
        self.lambdas = np.arange(400, 900, 5)  # lambda values from min to max, in nm
        # AR parameters
        self.dpix = d_ar / nbslab  # width in pixels
        assert self.dpix >= 10  # dpix < 10 is physically pointless
        # input and ouptut permittivities
        self.ep0 = 1
        self.epf = 9
        self.epmin = 1
        super().__init__(self._get_minimum_average_reflexion, ARCoatingVariable(nbslab, self.epmin, self.epf))
        self._descriptors.update(nbslab=nbslab, d_ar=d_ar)

    def _get_minimum_average_reflexion(self, x: np.ndarray) -> float:
        assert len(x) == self.dimension, f"Expected dimension {self.dimension}, got {len(x)}"
        if np.min(x) < self.epmin or np.max(x) > self.epf:  # acceptability
            return float('inf')
        value = 0.
        for lam in self.lambdas:
            RE = impedance_pix(x, self.dpix, lam, self.ep0, self.epf)  # only normal incidence
            value = value + RE / len(self.lambdas)
        return value
