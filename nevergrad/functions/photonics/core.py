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

import numpy as np
from nevergrad.parametrization import parameter as p
from . import photonics
from .. import base


def _make_parametrization(name: str, dimension: int, bounding_method: str = "bouncing", rolling: bool = False) -> p.Array:
    """Creates appropriate parametrization for a Photonics problem

    Parameters
    name: str
        problem name, among bragg, chirped and morpho
    dimension: int
        size of the problem among 16, 40 and 60 (morpho) or 80 (bragg and chirped)
    bounding_method: str
        transform type for the bounding ("arctan", "tanh", "bouncing" or "clipping"see `Array.bounded`)

    Returns
    -------
    Instrumentation
        the parametrization for the problem
    """
    if name == "bragg":
        shape = (2, dimension // 2)
        bounds = [(2, 3), (30, 180)]
    elif name == "chirped":
        shape = (1, dimension)
        bounds = [(30, 180)]
    elif name == "morpho":
        shape = (4, dimension // 4)
        bounds = [(0, 300), (0, 600), (30, 600), (0, 300)]
    else:
        raise NotImplementedError(f"Transform for {name} is not implemented")
    divisor = max(2, len(bounds))
    assert not dimension % divisor, f"points length should be a multiple of {divisor}, got {dimension}"
    assert shape[0] * shape[1] == dimension, f"Cannot work with dimension {dimension} for {name}: not divisible by {shape[0]}."
    b_array = np.array(bounds)
    assert b_array.shape[0] == shape[0]  # pylint: disable=unsubscriptable-object
    init = np.sum(b_array, axis=1, keepdims=True).dot(np.ones((1, shape[1],))) / 2
    array = p.Array(init=init)
    if bounding_method not in ("arctan", "tanh"):
        # sigma must be adapted for clipping and constraint methods
        sigma = p.Array(init=[[10.0]] if name != "bragg" else [[0.03], [10.0]]).set_mutation(exponent=2.0)  # type: ignore
        array.set_mutation(sigma=sigma)
    if rolling:
        array.set_mutation(custom=p.Choice(["gaussian", "cauchy", p.mutation.Translation(axis=1)]))
    array.set_bounds(b_array[:, [0]], b_array[:, [1]], method=bounding_method, full_range_sampling=True)
    array.set_recombination(p.mutation.Crossover(axis=1)).set_name("")
    assert array.dimension == dimension, f"Unexpected {array} for dimension {dimension}"
    return array


class Photonics(base.ExperimentFunction):
    """Function calling photonics code

    Parameters
    ----------
    name: str
        problem name, among bragg, chirped and morpho
    dimension: int
        size of the problem among 16, 40 and 60 (morpho) or 80 (bragg and chirped)
    transform: str
        transform type for the bounding ("arctan", "tanh", "bouncing" or "clipping", see `Array.bounded`)

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

    def __init__(self, name: str, dimension: int, bounding_method: str = "clipping", rolling: bool = False) -> None:
        assert name in ["bragg", "morpho", "chirped"]
        self.name = name
        self._base_func = {"morpho": photonics.morpho, "bragg": photonics.bragg, "chirped": photonics.chirped}[name]
        param = _make_parametrization(name=name, dimension=dimension, bounding_method=bounding_method, rolling=rolling)
        super().__init__(self._compute, param)
        self.register_initialization(name=name, dimension=dimension, bounding_method=bounding_method, rolling=rolling)
        self._descriptors.update(name=name, bounding_method=bounding_method, rolling=rolling)

    # pylint: disable=arguments-differ
    def evaluation_function(self, x: np.ndarray) -> float:  # type: ignore
        # pylint: disable=not-callable
        loss = self.function(x)
        base.update_leaderboard(f'{self.name},{self.parametrization.dimension}', loss, x, verbose=True)
        return loss

    def _compute(self, x: np.ndarray) -> float:
        x_cat = np.array(x, copy=False).ravel()
        assert x_cat.size == self.dimension
        try:
            output = self._base_func(x_cat)
        except Exception:  # pylint: disable=broad-except
            output = float("inf")
        if np.isnan(output):
            output = float("inf")
        return output
