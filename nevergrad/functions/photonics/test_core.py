# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import List
from unittest.mock import patch
import numpy as np
from ...common import testing
from . import core


@testing.parametrized(
    bragg=("bragg", [2.93, 2.18, 2.35, 2.12, 31.53, 15.98, 226.69, 193.11]),
    morpho=("morpho", [280.36, 52.96, 208.16, 72.69, 89.92, 60.37, 226.69, 193.11]),
    chirped=("chirped", [280.36, 52.96, 104.08, 36.34, 31.53, 15.98, 226.69, 193.11]),
)
def test_photonics_transforms(pb: str, expected: List[float]) -> None:
    with patch("shutil.which", return_value="here"):
        # dim 8 is easier to test... but it is actually not allowed. Nevermind here, HACK IT NEXT LINE
        with patch("nevergrad.functions.photonics.core._make_instrumentation", return_value=core._make_instrumentation(pb, 8)):
            func = core.Photonics(pb, 16)
    assert func.dimension == 8
    np.random.seed(24)
    x = np.random.normal(0, 1, size=8)
    all_x, _ = func.parametrization.data_to_arguments(x)
    output = np.concatenate(all_x)
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
def test_photonics_transforms_mean(pb: str, transform: str, expected: List[float]) -> None:
    with patch("shutil.which", return_value="here"):
        # dim 8 is easier to test... but it is actually not allowed. Nevermind here, HACK IT NEXT LINE
        with patch("nevergrad.functions.photonics.core._make_instrumentation", return_value=core._make_instrumentation(pb, 8, transform)):
            func = core.Photonics(pb, 16, transform=transform)
    all_x, _ = func.parametrization.data_to_arguments([0] * 8)
    output = np.concatenate(all_x)
    np.testing.assert_almost_equal(output, expected, decimal=2)


def test_morpho_transform_constraints() -> None:
    with patch("shutil.which", return_value="here"):
        func = core.Photonics("morpho", 60)
    x = np.random.normal(0, 5, size=60)  # std 5 to play with boundaries
    all_x, _ = func.parametrization.data_to_arguments(x)
    output = np.concatenate(all_x)
    assert np.all(output >= 0)
    q = len(x) // 4
    assert np.all(output[:q] <= 300)
    assert np.all(output[q: 3 * q] <= 600)
    assert np.all(output[2 * q: 3 * q] >= 30)
    assert np.all(output[3 * q:] <= 300)


def test_photonics_error() -> None:
    # check error
    photo = core.Photonics("bragg", 16)
    np.testing.assert_raises(AssertionError, photo, np.zeros(12).tolist())
    with warnings.catch_warnings(record=True) as w:
        output = photo(np.zeros(16))
        assert len(w) == 1
    np.testing.assert_almost_equal(output, float("inf"))


@testing.parametrized(
    morpho=("morpho", 100, 1.1647),
    chirped=("chirped", 150, 0.94439),
    bragg=("bragg", 2.5, 0.93216),
)
def test_photonics_values(name: str, value: float, expected: float) -> None:
    photo = core.Photonics(name, 16)
    np.testing.assert_almost_equal(photo(value * np.ones(16)), expected, decimal=4)
