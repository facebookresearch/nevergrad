# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This module has is based on code and ideas from:
# - Mamadou Aliou Barry
# - Marie-Claire Cambourieux
# - Rémi Pollès
# - Antoine Moreau
# from University Clermont Auvergne, CNRS, SIGMA Clermont, Institut Pascal.
#
# Publications:
# - Aliou Barry, Mamadou; Berthier, Vincent; Wilts, Bodo D.; Cambourieux, Marie-Claire; Pollès, Rémi;
#   Teytaud, Olivier; Centeno, Emmanuel; Biais, Nicolas; Moreau, Antoine (2018)
#   Evolutionary algorithms converge towards evolved biological photonic structures,
#   https://arxiv.org/abs/1808.04689
# - Defrance, J., Lemaître, C., Ajib, R., Benedicto, J., Mallet, E., Pollès, R., Plumey, J.-P.,
#   Mihailovic, M., Centeno, E., Ciracì, C., Smith, D.R. and Moreau, A., 2016.
#   Moosh: A Numerical Swiss Army Knife for the Optics of Multilayers in Octave/Matlab. Journal of Open Research Software, 4(1), p.e13.

import os
import shutil
from pathlib import Path
import numpy as np
from ...common.typetools import ArrayLike
from ... import instrumentation as inst
from ...instrumentation.transforms import TanhBound


class PhotonicsVariable(inst.var.utils.Variable[np.ndarray]):

    def __init__(self, name: str, dimension: int) -> None:
        self.name = name
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def data_to_argument(self, data: ArrayLike, deterministic: bool = True) -> np.ndarray:  # pylint: disable=unused-argument
        n = len(data)
        data = np.array(data, copy=False)
        assert not n % 4, f"points length should be a multiple of 4, got {n}"
        if self.name == "bragg":
            # n multiple of 2, from 16 to 80
            # domain (n=60): [2,3]^30 x [0,300]^30
            y = np.array(data, copy=True)
            y[:n // 2] = TanhBound(2, 3).forward(y[:n // 2])
            y[n // 2:] = TanhBound(0, 300).forward(y[n // 2:])
        elif self.name == "chirped":
            # n multiple of 2, from 10 to 80
            # domain (n=60): [0,300]^60
            y = TanhBound(0, 300).forward(data)
        elif self.name == "morpho":
            # n multiple of 4, from 16 to 60
            # domain (n=60): [0,300]^15 x [0,600]^15 x [30,600]^15 x [0,300]^15
            y = TanhBound(0, 1).forward(data)
            q = n // 4
            y[:q] *= 300
            y[q: 2 * q] *= 600
            y[2 * q: 3 * q] *= 570
            y[2 * q: 3 * q] += 30
            y[3 * q:] *= 300
        else:
            raise NotImplementedError(f"Transform for {self.name} is not implemented")
        return y

    def _short_repr(self) -> str:
        return "Photonics"


class Photonics(inst.InstrumentedFunction):
    """Function calling photonics code

    Parameters
    ----------
    pb: str
        problem name, among bragg, chirped and morpho
    dimension: int
        size of the problem among 16, 40 and 60 (morpho) or 80 (bragg and chirped)

    Returns
    -------
    float
        the fitness

    Notes
    -----
    - You will require an Octave installation (with conda: "conda install -c conda-forge octave" then re-source dfconda.sh)
    - Each function requires from around 1 to 5 seconds to compute
    - OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1 are enforced when spawning Octave because parallelization leads to
      deadlock issues here.

    Credit
    ------
    This module is based on code and ideas from:
    - Mamadou Aliou Barry
    - Marie-Claire Cambourieux
    - Rémi Pollès
    - Antoine Moreau
    from University Clermont Auvergne, CNRS, SIGMA Clermont, Institut Pascal.

    Publications
    ------------
    - Aliou Barry, Mamadou; Berthier, Vincent; Wilts, Bodo D.; Cambourieux, Marie-Claire; Pollès, Rémi;
      Teytaud, Olivier; Centeno, Emmanuel; Biais, Nicolas; Moreau, Antoine (2018)
      Evolutionary algorithms converge towards evolved biological photonic structures,
      https://arxiv.org/abs/1808.04689
    - Defrance, J., Lemaître, C., Ajib, R., Benedicto, J., Mallet, E., Pollès, R., Plumey, J.-P.,
      Mihailovic, M., Centeno, E., Ciracì, C., Smith, D.R. and Moreau, A. (2016)
      Moosh: A Numerical Swiss Army Knife for the Optics of Multilayers in Octave/Matlab. Journal of Open Research Software, 4(1), p.e13.
    """

    def __init__(self, name: str, dimension: int) -> None:
        if shutil.which("octave") is None:
            raise RuntimeError("Photonics function requires Octave to be installed in order to run")
        assert dimension in [16, 40, 60 if name == "morpho" else 80]
        assert name in ["bragg", "morpho", "chirped"]
        self.name = name
        path = Path(__file__).absolute().parent / 'src' / (name + '.m')
        assert path.exists(), f"Path {path} does not exist (anymore?)"
        self._func = inst.CommandFunction(["octave", "--no-history", "--norc", "--no-gui", "--quiet", path.name],
                                          cwd=path.parent, verbose=False,
                                          env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
        super().__init__(self._compute, PhotonicsVariable(name=name, dimension=dimension))
        self._descriptors.update(name=name)

    def _compute(self, x: np.ndarray) -> float:
        assert len(x) == self.dimension, f"Got length {len(x)} but expected {self.dimension}"
        output = self._func(*x.tolist())
        output_list = output.strip().splitlines()
        try:
            value = float(output_list[-1])
        except Exception as e:  # pylint: disable=bare-except
            raise RuntimeError(f'Could not parse output "{output}"') from e
        return value
