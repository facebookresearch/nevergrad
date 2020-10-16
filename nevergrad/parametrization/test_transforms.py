# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import pytest
import numpy as np
from ..common import testing
from . import transforms


@testing.parametrized(
    affine=(transforms.Affine(3, 4), "Af(3,4)"),
    reverted=(transforms.Affine(3, 4).reverted(), "Rv(Af(3,4))"),
    exponentiate=(transforms.Exponentiate(3, 4), "Ex(3,4)"),
    tanh=(transforms.TanhBound(3.0, 4.5), "Th(3,4.5)"),
    arctan=(transforms.ArctanBound(3, 4), "At(3,4)"),
    cumdensity=(transforms.CumulativeDensity(), "Cd(0,1)"),
    cumdensity2=(transforms.CumulativeDensity(1, 3), "Cd(1,3)"),
    clipping=(transforms.Clipping(None, 1e12), "Cl(None,1000000000000)"),
    bouncing=(transforms.Clipping(-12000, 12000, bounce=True), "Cl(-12000,12000,b)"),
    fourrier=(transforms.Fourrier(), "F(0)"),
)
def test_back_and_forth(transform: transforms.Transform, string: str) -> None:
    x = np.random.normal(0, 1, size=12)
    y = transform.forward(x)
    x2 = transform.backward(y)
    np.testing.assert_array_almost_equal(x2, x)
    np.testing.assert_equal(transform.name, string)


@testing.parametrized(
    affine=(transforms.Affine(3, 4), [0, 1, 2], [4, 7, 10]),
    reverted=(transforms.Affine(3, 4).reverted(), [4, 7, 10], [0, 1, 2]),
    exponentiate=(transforms.Exponentiate(10, -1.), [0, 1, 2], [1, .1, .01]),
    tanh=(transforms.TanhBound(3, 5), [-100000, 100000, 0], [3, 5, 4]),
    arctan=(transforms.ArctanBound(3, 5), [-100000, 100000, 0], [3, 5, 4]),
    bouncing=(transforms.Clipping(0, 10, bounce=True), [-1, 22, 3], [1, 0, 3]),
    cumdensity=(transforms.CumulativeDensity(), [-10, 0, 10], [0, .5, 1]),
    cumdensity_bounds=(transforms.CumulativeDensity(2, 4), [-10, 0, 10], [2, 3, 4]),
)
def test_vals(transform: transforms.Transform, x: tp.List[float], expected: tp.List[float]) -> None:
    y = transform.forward(np.array(x))
    np.testing.assert_almost_equal(y, expected, decimal=5)


@testing.parametrized(
    tanh=(transforms.TanhBound(0, 5), [2, 4], None),
    tanh_err=(transforms.TanhBound(0, 5), [2, 4, 6], ValueError),
    clipping=(transforms.Clipping(0), [2, 4, 6], None),
    clipping_err=(transforms.Clipping(0), [-2, 4, 6], ValueError),
    arctan=(transforms.ArctanBound(0, 5), [2, 4, 5], None),
    arctan_err=(transforms.ArctanBound(0, 5), [-1, 4, 5], ValueError),
    cumdensity=(transforms.CumulativeDensity(), [0, .5], None),
    cumdensity_err=(transforms.CumulativeDensity(), [-0.1, .5], ValueError),
)
def test_out_of_bound(transform: transforms.Transform, x: tp.List[float], expected: tp.Optional[tp.Type[Exception]]) -> None:
    if expected is None:
        transform.backward(np.array(x))
    else:
        with pytest.raises(expected):
            transform.backward(np.array(x))


@testing.parametrized(
    tanh=(transforms.TanhBound, [1., 100.]),
    arctan=(transforms.ArctanBound, [0.9968, 99.65]),
    clipping=(transforms.Clipping, [1, 90]),
)
def test_multibounds(transform_cls: tp.Type[transforms.BoundTransform], expected: tp.List[float]) -> None:
    transform = transform_cls([0, 0], [1, 100])
    output = transform.forward(np.array([100, 90]))
    np.testing.assert_almost_equal(output, expected, decimal=2)
    # shapes
    with pytest.raises(ValueError):
        transform.forward(np.array([-3, 5, 4]))
    with pytest.raises(ValueError):
        transform.backward(np.array([-3, 5, 4]))
    # bound error
    with pytest.raises(ValueError):
        transform_cls([0, 0], [0, 100])
    # two Nones
    with pytest.raises(ValueError):
        transform_cls(None, None)


@testing.parametrized(
    both_sides=(transforms.Clipping(0, 1), [0, 1.]),
    one_side=(transforms.Clipping(a_max=1), [-3, 1.]),
)
def test_clipping(transform: transforms.Transform, expected: tp.List[float]) -> None:
    y = transform.forward(np.array([-3, 5]))
    np.testing.assert_array_equal(y, expected)
