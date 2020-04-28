# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch
import numpy as np
import pandas as pd
from ...common import testing
from . import problems


def test_kmeans_distance() -> None:
    centers = np.array([[1, 2, 3, 4], [3, 4, 5, 6]])
    points = np.array([[1, 2, 3, 4], [3, 4, 5, 6], [1, 2, 5, 6]])
    output = problems._kmeans_distance(points, centers)
    np.testing.assert_equal(output, 8)


def test_clustering() -> None:
    data = np.random.normal(size=(20, 2))
    num_clusters = 5
    with patch("nevergrad.functions.mlda.datasets.get_data") as data_getter:
        data_getter.return_value = data
        func = problems.Clustering.from_mlda(name="Ruspini", num_clusters=num_clusters, rescale=True).copy()
        np.testing.assert_equal(func.dimension, 10)
    func(np.arange(10).reshape((num_clusters, -1)))
    instru_str = "Array{(5,2)}"
    testing.printed_assert_equal(func.descriptors,
                                 {"parametrization": instru_str, "function_class": "Clustering", "dimension": 10,
                                  "name": "Ruspini", "num_clusters": 5, "rescale": True})


def test_compute_perceptron() -> None:
    p = np.random.normal(size=10)
    data = np.random.normal(size=(5, 2))
    square_sum = 0
    # compute explicitely the article function:
    # g(x) = v0+v1tanh(w01+x·w11)+v2tanh(w02+x·w12)+v3tanh(w03+x·w13)
    for x, y in data:
        z = p[-1]
        for k in range(3):
            z += p[6 + k] * np.tanh(p[3 + k] + p[k] * x)
        square_sum += (z - y)**2
    output = problems.Perceptron(data[:, 0], data[:, 1]).copy()(p)
    np.testing.assert_almost_equal(output, square_sum / 5)


def test_perceptron() -> None:
    func = problems.Perceptron.from_mlda(name="quadratic").copy()
    output = func([k for k in range(10)])
    np.testing.assert_almost_equal(output, 876.837, decimal=4)
    np.testing.assert_equal(func.descriptors["name"], "quadratic")


@testing.parametrized(
    virus=("Virus",),
    employees=("Employees",),
)
def test_sammon_mapping(name: str) -> None:
    data = np.arange(6).reshape(3, 2) if name == "Virus" else pd.DataFrame(data=np.arange(12).reshape(3, 4))
    with patch("nevergrad.functions.mlda.datasets.get_data") as data_getter:
        data_getter.return_value = data
        func = problems.SammonMapping.from_mlda(name=name).copy()
    value = func(np.arange(6).reshape((-1, 2)))
    np.testing.assert_almost_equal(value, 0 if name == "Virus" else 5.152, decimal=4)


def test_sammon_circle() -> None:
    func = problems.SammonMapping.from_2d_circle().copy()
    assert np.max(func._proximity) <= 2.


def test_landscape() -> None:
    data = np.arange(6).reshape(3, 2)
    with patch("nevergrad.functions.mlda.datasets.get_data") as data_getter:
        data_getter.return_value = data
        func = problems.Landscape(transform=None).copy()
        sfunc = problems.Landscape(transform="square")
    np.testing.assert_equal(func(0, 0), 5)
    np.testing.assert_equal(func(-.2, -0.2), 5)
    np.testing.assert_equal(func(-.6, -0.2), float("inf"))
    np.testing.assert_equal(func(2, 1), 0)
    np.testing.assert_equal(func(2.6, 1), float("inf"))
    # with square
    args = sfunc.parametrization.spawn_child().set_standardized_data([-1, -1]).args  # bottom left
    np.testing.assert_equal(args, [0, 0])
    args = sfunc.parametrization.spawn_child().set_standardized_data([1, 1]).args  # upper right
    np.testing.assert_equal(args, [2, 1])


def test_landscape_gaussian() -> None:
    data = np.arange(6).reshape(3, 2)
    with patch("nevergrad.functions.mlda.datasets.get_data") as data_getter:
        data_getter.return_value = data
        func = problems.Landscape(transform="gaussian").copy()
    output = func.parametrization.spawn_child().set_standardized_data([-144, -144]).args
    np.testing.assert_equal(output, [0, 0])  # should be mapped to 0, 0
    output = func.parametrization.spawn_child().set_standardized_data([144, 144]).args
    np.testing.assert_array_equal(output, [2, 1])  # last element
    testing.printed_assert_equal(func.descriptors, {"parametrization": "gaussian", "function_class": "Landscape", "dimension": 2})
