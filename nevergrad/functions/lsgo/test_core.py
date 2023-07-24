# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Callable, Optional, List, Iterable, Type, Union
import numpy as np
import pytest
from nevergrad.common import testing
from . import _core


@testing.parametrized(
    sphere=(_core.sphere, 9.043029, None),
    elliptic=(_core.elliptic, 262389.541996, None),
    ackley=(_core.ackley, 4.7298473, None),
    schwefel=(_core.schwefel_1_2, 128.709765, None),
    rosenbrock=(_core.rosenbrock, 1967.57859, None),
    sphere_m=(_core.sphere, 30, [1, 2, 3, 4]),
    elliptic_m=(_core.elliptic, 16090401, [1, 2, 3, 4]),
    ackley_m=(_core.ackley, 2.254444, [0.1, 0.2, 0.3]),
    schwefel_m=(_core.schwefel_1_2, 146, [1, 2, 3, 4]),
    rosenbrock_m=(_core.rosenbrock, 2705, [1, 2, 3, 4]),
)
def test_core_function(
    func: Callable[[np.ndarray], float], expected: float, data: Optional[List[float]]
) -> None:
    if data is None:
        data = [
            0.662,
            -0.217,
            -0.968,
            1.867,
            0.101,
            0.575,
            0.199,
            1.576,
            1.006,
            0.182,
            -0.092,
            0.466,
        ]  # only for non-regression
    value = func(np.array(data))
    np.testing.assert_almost_equal(value, expected, decimal=5)


@testing.parametrized(
    irregularity=(
        _core.irregularity,
        [-10, -2, -1, 0, 0.1, 1, 2, 2.5],
        [-10.426, -2.021, -1, 0, 0.107, 1, 1.988, 2.635],
    ),
    asymmetry_04=(
        _core.Asymmetry(0.2),
        [-10, -2, -1, 0, 0.1, 1, 2, 2.5],
        [-10, -2, -1, 0, 0.092, 1, 2.366, 3.34],
    ),
    asymmetry_02=(_core.Asymmetry(0.4), [-1, 0, 0.1, 1, 2], [-1, 0, 0.086, 1, 2.96]),
    illconditionning_10=(
        _core.Illconditionning(10.0),
        [-2, -1, 0, 0.1, 1, 2],
        [-2.0, -1.259, 0.0, 0.2, 2.512, 6.325],
    ),
    illconditionning_1=(
        _core.Illconditionning(1.0),
        [-2, -1, 0, 0.1, 1, 2],
        [-2, -1, 0, 0.1, 1, 2],
    ),  # no change expected
    translation=(_core.Translation(np.array([1, 2])), [0, 10], [-1, 8]),
    rotation=(_core.Rotation(np.array([[0, 1], [1, 0]])), [0, 10], [10, 0]),
)
def test_transform(
    func: Callable[[np.ndarray], np.ndarray], data: List[float], expected: List[float]
) -> None:
    value = func(np.array(data))
    np.testing.assert_almost_equal(value, expected, decimal=3)


@testing.parametrized(
    simple=(range(8), [3, 3, 2], 0, [[0, 1, 2], [3, 4, 5], [6, 7]]),
    bad_dim=(range(8), [3, 3, 4], 0, ValueError),
    bad_dim_overlap=(range(6), [3, 3, 1], 1, ValueError),
    overlap=(np.arange(6), [3, 3, 2], 1, [[0, 1, 2], [2, 3, 4], [4, 5]]),
    permut=([0, 2, 3, 5, 1, 4], [3, 2, 3], 1, [[0, 2, 3], [3, 5], [5, 1, 4]]),
    wrong_start=(1 + np.arange(6), [3, 3, 2], 1, AssertionError),
)
def test_split(
    permutation: Iterable[float],
    dimensions: List[int],
    overlap: int,
    expected: Union[Type[Exception], List[np.ndarray]],
) -> None:
    perm = np.array(list(permutation))
    if isinstance(expected, list):
        output = _core.split(perm, dimensions, overlap)
        np.testing.assert_equal(output, expected)
    else:
        with pytest.raises(expected if not isinstance(expected, list) else None):
            _core.split(perm, dimensions, overlap)
