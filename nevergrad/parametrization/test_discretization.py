# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List
import warnings
import numpy as np
from ..common import testing
from . import discretization


@testing.parametrized(
    arity2=(2, [0, -4, 0, 4, 0, 0], [0, 1, .5], [0, 1, 0]),
    arity2_1=(2, [0, 40], [1], [1]),
    arity3=(3, [0, -4, 0, 0, 4, 0], [1, 1], [0, 1]),  # first is 0 or 2, second is 1
    arity2_0_sum=(2, [0, 0], [.5], [0]),  # first is 0 or 2, second is 1
    pinf_case=(2, [0, np.inf], [1], [1]),
    nan_case=(2, [np.nan, 0], [1], [1]),
    ninf_case=(2, [-np.inf, 0], [1], [1]),
    all_ninf_case=(2, [-np.inf, -np.inf], [.5], [0]),
)
def test_softmax_discretization(arity: int, data: List[float], expected: List[float],
                                deterministic_expected: List[float]) -> None:
    coeffs = np.array(data, copy=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        output = np.mean([discretization.softmax_discretization(coeffs, arity=arity) for _ in range(1000)], axis=0)
    np.testing.assert_array_equal(coeffs, data, err_msg="Input data was modified")
    np.testing.assert_almost_equal(output, expected, decimal=1, err_msg="Wrong mean value")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        deterministic_output = discretization.softmax_discretization(coeffs, arity=arity, random=False)
    np.testing.assert_array_equal(deterministic_output, deterministic_expected, err_msg="Wrong deterministic value")


@testing.parametrized(
    arity2=(2, [.1, -2, 0], [1, 0, 0]),
    arity100=(100, [-15, -2, -.3, .3, 2, 15], [0, 2, 38, 61, 97, 99]),
    borderline_cases_100=(100, [-np.inf, np.inf, np.nan], [0, 99, 0]),
    borderline_cases_2=(2, [-np.inf, np.inf, np.nan], [0, 1, 0]),
)
def test_thresholding_discretization(arity: int, data: List[float], expected: List[float]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        output = discretization.threshold_discretization(data, arity)
    np.testing.assert_array_equal(output, expected)


def test_inverse_threshold_discretization() -> None:
    arity = 4
    indexes = np.arange(arity)  # Test all possible indexes
    data = discretization.inverse_threshold_discretization(indexes, arity)
    np.testing.assert_array_equal(discretization.threshold_discretization(data, arity), indexes)


def test_inverse_softmax_discretization() -> None:
    output = discretization.inverse_softmax_discretization(arity=5, index=2)
    np.testing.assert_array_almost_equal(output, [0, 0, 0.539, 0, 0], decimal=5)
