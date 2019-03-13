# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from . import core


def test_impedence_pix() -> None:
    ep0 = 1
    epf = 9
    x = np.array([3., 5, 1, 9])
    dpix = 37
    lam = 400
    output = core.impedance_pix(x, dpix, lam, ep0, epf)
    np.testing.assert_almost_equal(output, 46.64, decimal=2)


def test_arcoating_reflexion_function() -> None:
    func = core.ARCoating(nbslab=4)
    output = func._get_minimum_average_reflexion(np.array([4.56386701, 5.65210553, 6.24006888, 7.18400555]))
    # np.testing.assert_almost_equal(output, 13.320815699203614)  # Before change
    np.testing.assert_almost_equal(output, 12.702, decimal=3)


def test_arcoating_transform_and_call() -> None:
    nbslab = 8
    func = core.ARCoating(nbslab=nbslab)
    x = 7 * np.random.normal(size=nbslab)  # make sure it touches space boundaries
    value = func(x)  # should not touch boundaries, so value should be < np.inf
    assert value < np.inf
    value = func(np.arange(8))  # should not touch boundaries, so value should be < np.inf
    np.testing.assert_almost_equal(value, 11.31129)
