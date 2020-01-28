# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import typing as tp
import pytest
import numpy as np
from ...common import testing
from . import core


@testing.parametrized(
    bragg=("bragg", [2.93, 2.18, 2.35, 2.12, 31.53, 15.98, 226.69, 193.11]),
    morpho=("morpho", [280.36, 52.96, 208.16, 72.69, 89.92, 60.37, 226.69, 193.11]),
    chirped=("chirped", [280.36, 52.96, 104.08, 36.34, 31.53, 15.98, 226.69, 193.11]),
)
def test_photonics_bounding_methods(pb: str, expected: tp.List[float]) -> None:
    func = core.Photonics(pb, 8, bounding_method="tanh")
    np.random.seed(24)
    x = np.random.normal(0, 1, size=8)
    # param = func.parametrization.spawn_child()
    # # legacy test, now sigma has been modified, we revert this in the following line
    # func.parametrization = param.set_mutation(sigma=np.ones(param.sigma.value.shape))  # type: ignore
    output = func.parametrization.spawn_child().set_standardized_data(x).value.ravel()
    np.testing.assert_almost_equal(output, expected, decimal=2)


@testing.parametrized(
    # bragg domain (n=60): [2,3]^30 x [0,300]^30
    bragg_tanh=("bragg", "tanh", [2.5, 2.5, 2.5, 2.5, 150., 150., 150., 150.]),
    bragg_arctan=("bragg", "arctan", [2.5, 2.5, 2.5, 2.5, 150., 150., 150., 150.]),
    # chirped domain (n=60): [0,300]^60
    chirped_tanh=("chirped", "tanh", [150., 150., 150., 150., 150., 150., 150., 150.]),
    chirped_arctan=("chirped", "arctan", [150., 150., 150., 150., 150., 150., 150., 150.]),
    # morpho domain (n=60): [0,300]^15 x [0,600]^15 x [30,600]^15 x [0,300]^15
    morpho_tanh=("morpho", "tanh", [150., 150., 300., 300., 315., 315., 150., 150.]),
    morpho_arctan=("morpho", "arctan", [150., 150., 300., 300., 315., 315., 150., 150.]),
)
def test_photonics_bounding_methods_mean(pb: str, bounding_method: str, expected: tp.List[float]) -> None:
    func = core.Photonics(pb, 8, bounding_method=bounding_method)
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


def test_photonics_recombination() -> None:
    func = core.Photonics("chirped", 16)
    func.parametrization.random_state.seed(24)
    array = func.parametrization.spawn_child()
    array.value = 12 * np.ones(array.value.shape)
    array.recombine(func.parametrization)
    expected = [12] * 5 + [150, 150, 12]
    np.testing.assert_array_equal(array.value, np.ones((2, 1)).dot(np.array(expected)[None, :]))


def test_photonics_error() -> None:
    # check error
    photo = core.Photonics("bragg", 16)
    np.testing.assert_raises(AssertionError, photo, np.zeros(12))
    with warnings.catch_warnings(record=True) as w:
        output = photo(np.zeros(16))
        assert len(w) == 1
    np.testing.assert_almost_equal(output, float("inf"))


@pytest.mark.parametrize("method", ["clipping", "arctan", "tanh", "constraint"])  # type: ignore
@pytest.mark.parametrize("name", ["bragg", "morpho", "chirped"])
def test_no_warning(name: str, method: str):
    with warnings.catch_warnings(record=True) as w:
        core.Photonics(name, 24, bounding_method=method)
        assert not w, f"Got a warning at initialization: {w[0]}"


@testing.parametrized(
    morpho=("morpho", 100, 1.1647),
    chirped=("chirped", 150, 0.94439),
    bragg=("bragg", 2.5, 0.93216),
)
def test_photonics_values(name: str, value: float, expected: float) -> None:
    photo = core.Photonics(name, 16)
    np.testing.assert_almost_equal(photo(value * np.ones(16)), expected, decimal=4)
