# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
    np.random.seed(24)
    with patch("shutil.which", return_value="here"):
        func = core.Photonics(pb, 16)  # should be 8... but it is actually not allowed. Nevermind here, HACK IT NEXT LINE
    func.instrumentation.args[0]._dimension = 8  # type: ignore
    x = np.random.normal(0, 1, size=8)
    (output,), _ = func.instrumentation.data_to_arguments(x)
    np.testing.assert_almost_equal(output, expected, decimal=2)
    np.random.seed(24)
    x2 = np.random.normal(0, 1, size=8)
    np.testing.assert_almost_equal(x, x2, decimal=2, err_msg="x was modified in the process")


def test_morpho_transform_constraints() -> None:
    with patch("shutil.which", return_value="here"):
        func = core.Photonics("morpho", 60)
    x = np.random.normal(0, 5, size=60)  # std 5 to play with boundaries
    (output,), _ = func.instrumentation.data_to_arguments(x)
    assert np.all(output >= 0)
    q = len(x) // 4
    assert np.all(output[:q] <= 300)
    assert np.all(output[q: 3 * q] <= 600)
    assert np.all(output[2 * q: 3 * q] >= 30)
    assert np.all(output[3 * q:] <= 300)


def test_photonics() -> None:
    with patch("shutil.which", return_value="here"):
        photo = core.Photonics("bragg", 16)
    with patch("nevergrad.instrumentation.utils.CommandFunction.__call__", return_value="line1\n12\n"):
        with patch("nevergrad.instrumentation.utils.CommandFunction.__call__", return_value="line1\n12\n"):
            output = photo(np.zeros(16))
    np.testing.assert_equal(output, 12)
    # check error
    with patch("nevergrad.instrumentation.utils.CommandFunction.__call__", return_value="line1\n"):
        np.testing.assert_raises(RuntimeError, photo, np.zeros(16).tolist())
    np.testing.assert_raises(AssertionError, photo, np.zeros(12).tolist())
