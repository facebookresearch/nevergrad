# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from unittest import SkipTest
import warnings
import typing as tp
import pytest
import numpy as np
from nevergrad.common import testing
from . import core


@testing.parametrized(
    bragg=("bragg", [2.93, 2.18, 2.35, 2.12, 45.77, 37.99, 143.34, 126.55]),
    morpho=("morpho", [280.36, 52.96, 208.16, 72.69, 89.92, 60.37, 226.69, 193.11]),
    chirped=("chirped", [170.18, 56.48, 82.04, 48.17, 45.77, 37.99, 143.34, 126.55])
)
def test_photonics_bounding_methods(pb: str, expected: tp.List[float]) -> None:
    func = core.Photonics(pb, 8, bounding_method="tanh")
    np.random.seed(24)
    x = np.random.normal(0, 1, size=8)
    output = func.parametrization.spawn_child().set_standardized_data(x).value.ravel()
    np.testing.assert_almost_equal(output, expected, decimal=2)


@testing.parametrized(
    # bragg domain (n=60): [2,3]^30 x [30,180]^30
    bragg=("bragg", [2.5, 2.5, 2.5, 2.5, 105., 105., 105., 105.]),
    # chirped domain (n=60): [30,170]^60
    chirped=("chirped", [105., 105., 105., 105., 105., 105., 105., 105.]),
    # morpho domain (n=60): [0,300]^15 x [0,600]^15 x [30,600]^15 x [0,300]^15
    morpho=("morpho", [150., 150., 300., 300., 315., 315., 150., 150.]),
)
def test_photonics_mean(pb: str, expected: tp.List[float]) -> None:
    func = core.Photonics(pb, 8)
    all_x = func.parametrization.value
    output = all_x.ravel()
    np.testing.assert_almost_equal(output, expected, decimal=2)


def test_morpho_bounding_method_constraints() -> None:
    func = core.Photonics("morpho", 60, bounding_method="arctan")
    x = np.random.normal(0, 5, size=60)  # std 5 to play with boundaries
    output1 = func.parametrization.spawn_child().set_standardized_data(x)
    output2 = output1.sample()
    for output in (output1, output2):
        assert np.all(output.value >= 0)
        assert np.all(output.value[[0, 3], :] <= 300)
        assert np.all(output.value[[1, 2], :] <= 600)
        assert np.all(output.value[2, :] >= 30)


def test_photonics_bragg_recombination() -> None:
    func = core.Photonics("bragg", 8)
    # func.parametrization.set_recombination(ng.p.mutation.RavelCrossover())  # type: ignore
    func.parametrization.random_state.seed(24)
    arrays = [func.parametrization.spawn_child() for _ in range(2)]
    arrays[0].value = [[2, 2, 2, 2], [35, 35, 35, 35]]
    arrays[1].value = [[3, 3, 3, 3], [45, 45, 45, 45]]
    arrays[0].recombine(arrays[1])
    expected = [[3, 2, 2, 2], [45, 35, 35, 35]]
    np.testing.assert_array_equal(arrays[0].value, expected)


def test_photonics_custom_mutation() -> None:
    func = core.Photonics("morpho", 16, rolling=True)
    param = func.parametrization.spawn_child()
    for _ in range(10):
        param.mutate()


def test_photonics_error() -> None:
    # check error
    photo = core.Photonics("bragg", 16)
    np.testing.assert_raises(AssertionError, photo, np.zeros(12))
    with warnings.catch_warnings(record=True) as ws:
        output = photo(np.zeros(16))
        # one warning on Ubuntu, two warnings with Windows
        assert any(isinstance(w.message, RuntimeWarning) for w in ws)
    np.testing.assert_almost_equal(output, float("inf"))


@pytest.mark.parametrize("method", ["clipping", "arctan", "tanh", "constraint"])  # type: ignore
@pytest.mark.parametrize("name", ["bragg", "morpho", "chirped"])  # type: ignore
def test_no_warning(name: str, method: str) -> None:
    with warnings.catch_warnings(record=True) as w:
        core.Photonics(name, 24, bounding_method=method)
        assert not w, f"Got a warning at initialization: {w[0]}"


@testing.parametrized(
    # morpho=("morpho", 100, 1.1647),  # too slow
    chirped=("chirped", 150, 0.94439),
    bragg=("bragg", 2.5, 0.93216),
)
def test_photonics_values(name: str, value: float, expected: float) -> None:
    if name == "morpho" and os.environ.get("CIRCLECI", False):
        raise SkipTest("Too slow in CircleCI")
    photo = core.Photonics(name, 16)
    np.testing.assert_almost_equal(photo(value * np.ones(16)), expected, decimal=4)
    np.testing.assert_almost_equal(photo.evaluation_function(value * np.ones(16)), expected, decimal=4)


GOOD_CHIRPED = [89.04887416, 109.54188095, 89.74520725, 121.81700431,
                179.99830918, 124.38222473, 95.31017129, 116.0239629,
                92.92345776, 118.06108198, 179.99965859, 116.89288181,
                88.90191494, 110.30816229, 93.11974992, 137.42629858,
                118.81810084, 110.74139708, 85.15270955, 100.9382438,
                81.44070951, 100.6382896, 84.97336252, 110.59252719,
                134.89164276, 121.84205195, 89.28450356, 106.72776991,
                85.77168797, 102.33562547]


@testing.parametrized(
    morpho=("morpho", 1.127904, None),
    chirped=("chirped", 0.594587, None),
    good_chirped=("chirped", 0.275923, GOOD_CHIRPED),  # supposed to be better
    bragg=("bragg", 0.96776, None)
)
def test_photonics_values_random(name: str, expected: float, data: tp.Optional[tp.List[float]]) -> None:
    if name == "morpho" and os.environ.get("CIRCLECI", False):
        raise SkipTest("Too slow in CircleCI")
    size = len(data) if data is not None else (16 if name != "morpho" else 4)
    photo = core.Photonics(name, size)
    if data is None:
        x = np.random.RandomState(12).normal(0, 1, size=size)
        candidate = photo.parametrization.spawn_child().set_standardized_data(x)
    else:
        candidate = photo.parametrization.spawn_child(new_value=[data])
    for func in [photo, photo.evaluation_function]:
        np.testing.assert_almost_equal(func(candidate.value), expected, decimal=4)  # type: ignore
