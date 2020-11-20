# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import numpy as np
import nevergrad as ng
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


def test_arcoating_recombination() -> None:
    func = core.ARCoating(nbslab=6)
    func.parametrization.random_state.seed(24)
    arrays: tp.List[ng.p.Array] = []
    for num in [3, 5]:
        arrays.append(func.parametrization.spawn_child())  # type: ignore
        arrays[-1].value = num * np.ones(arrays[0].value.shape)
    arrays[0].recombine(arrays[1])
    expected = [3., 3., 3., 5., 3., 3.]
    np.testing.assert_array_equal(arrays[0].value, expected)


def test_arcoating_transform_and_call() -> None:
    nbslab = 8
    func = core.ARCoating(nbslab=nbslab)
    x = 7 * np.random.normal(size=nbslab)  # make sure it touches space boundaries
    data = func.parametrization.spawn_child().set_standardized_data(x).args[0]
    value = func(data)  # should not touch boundaries, so value should be < np.inf
    assert value < np.inf
    data = func.parametrization.spawn_child().set_standardized_data(np.arange(8)).args[0]
    for f in [func, func.evaluation_function]:
        np.testing.assert_almost_equal(f(data), 16.5538936)  # type: ignore
