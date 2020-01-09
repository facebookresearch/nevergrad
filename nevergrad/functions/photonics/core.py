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
from typing import List
from pathlib import Path
import numpy as np
from ..base import ExperimentFunction
from ... import instrumentation as inst


def _make_instrumentation(name: str, dimension: int, transform: str = "tanh") -> inst.Instrumentation:
    """Creates appropriate instrumentation for a Photonics problem

    Parameters
    name: str
        problem name, among bragg, chirped and morpho
    dimension: int
        size of the problem among 16, 40 and 60 (morpho) or 80 (bragg and chirped)
    transform: str
        transform type for the bounding ("arctan", "tanh" or "clipping", see `Array.bounded`)

    Returns
    -------
    Instrumentation
        the instrumentation for the problem
    """
    assert not dimension % 4, f"points length should be a multiple of 4, got {dimension}"
    n = dimension // 4
    arrays: List[inst.var.Array] = []
    if name == "bragg":
        # n multiple of 2, from 16 to 80
        # domain (n=60): [2,3]^30 x [0,300]^30
        arrays.extend([inst.var.Array(n).bounded(2, 3, transform=transform) for _ in range(2)])
        arrays.extend([inst.var.Array(n).bounded(0, 300, transform=transform) for _ in range(2)])
    elif name == "chirped":
        # n multiple of 2, from 10 to 80
        # domain (n=60): [0,300]^60
        arrays = [inst.var.Array(n).bounded(0, 300, transform=transform) for _ in range(4)]
    elif name == "morpho":
        # n multiple of 4, from 16 to 60
        # domain (n=60): [0,300]^15 x [0,600]^15 x [30,600]^15 x [0,300]^15
        arrays.extend([inst.var.Array(n).bounded(0, 300, transform=transform),
                       inst.var.Array(n).bounded(0, 600, transform=transform),
                       inst.var.Array(n).bounded(30, 600, transform=transform),
                       inst.var.Array(n).bounded(0, 300, transform=transform)])
    else:
        raise NotImplementedError(f"Transform for {name} is not implemented")
    instrumentation = inst.Instrumentation(*arrays)
    assert instrumentation.dimension == dimension
    return instrumentation


class Photonics(ExperimentFunction):
    """Function calling photonics code

    Parameters
    ----------
    name: str
        problem name, among bragg, chirped and morpho
    dimension: int
        size of the problem among 16, 40 and 60 (morpho) or 80 (bragg and chirped)
    transform: str
        transform type for the bounding ("arctan", "tanh" or "clipping", see `Array.bounded`)

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

    def __init__(self, name: str, dimension: int, transform: str = "tanh") -> None:
        if shutil.which("octave") is None:
            raise RuntimeError("Photonics function requires Octave to be installed in order to run")
        assert dimension in [8, 16, 40, 60 if name == "morpho" else 80]
        assert name in ["bragg", "morpho", "chirped"]
        self.name = name
        path = Path(__file__).absolute().parent / 'src' / (name + '.m')
        assert path.exists(), f"Path {path} does not exist (anymore?)"
        self._func = inst.CommandFunction(["octave-cli", "--no-gui", "--no-history", "--norc", "--quiet", "--no-window-system", path.name],
                                          cwd=path.parent, verbose=False,
                                          env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
        super().__init__(self._compute, _make_instrumentation(name=name, dimension=dimension, transform=transform))
        self.register_initialization(name=name, dimension=dimension, transform=transform)
        self._descriptors.update(name=name)

    def _compute(self, *x: np.ndarray) -> float:
        x_cat = np.concatenate(x)
        assert x_cat.shape == (self.dimension,), f"Got length {x_cat.shape} but expected ({self.dimension},)"
        output = self._func(*x_cat.tolist())
        output_list = output.strip().splitlines()
        try:
            value = float(output_list[-1])
        except Exception as e:  # pylint: disable=bare-except
            raise RuntimeError(f'Could not parse output "{output}"') from e
        return value
