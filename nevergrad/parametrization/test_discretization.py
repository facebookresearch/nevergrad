# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import warnings
import numpy as np
from ..common import testing
from . import discretization


@testing.parametrized(
    arity2=(2, [0.1, -2, 0], [1, 0, 0]),
    arity100=(100, [-15, -2, -0.3, 0.3, 2, 15], [0, 2, 38, 61, 97, 99]),
    borderline_cases_100=(100, [-np.inf, np.inf, np.nan], [0, 99, 0]),
    borderline_cases_2=(2, [-np.inf, np.inf, np.nan], [0, 1, 0]),
)
def test_thresholding_discretization(arity: int, data: tp.List[float], expected: tp.List[float]) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        output = discretization.threshold_discretization(data, arity)
    np.testing.assert_array_equal(output, expected)


def test_inverse_threshold_discretization() -> None:
    arity = 4
    indexes = np.arange(arity)  # Test all possible indexes
    data = discretization.inverse_threshold_discretization(indexes, arity)
    np.testing.assert_array_equal(discretization.threshold_discretization(data, arity), indexes)


def test_encoder_probabilities() -> None:
    weights = np.array(
        [
            [0, 0, 0],
            [0, 0, 100],
            [np.nan, 0, 1],
            [np.inf, np.inf, np.inf],
            [np.inf, np.inf, 12],
            [0, -np.inf, 0],
        ]
    )
    rng = np.random.RandomState(12)
    enc = discretization.Encoder(weights, rng=rng)
    proba = enc.probabilities()
    expected = [
        [0.333, 0.333, 0.333],
        [0, 0, 1],
        [0, 0.269, 0.731],
        [0.333, 0.333, 0.333],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
    ]
    np.testing.assert_array_almost_equal(proba, expected, decimal=3)


def test_encoder() -> None:
    weights = np.array([[0, 0, 0], [0, 0, 100]])
    rng = np.random.RandomState(14)
    enc = discretization.Encoder(weights, rng=rng)
    np.testing.assert_equal(enc.encode(), [1, 2])
    np.testing.assert_equal(enc.encode(), [2, 2])
    #
    for _ in range(10):
        np.testing.assert_equal(enc.encode(True), [0, 2])
