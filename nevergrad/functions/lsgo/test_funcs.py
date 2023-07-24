# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Any, Callable
import numpy as np
from nevergrad.common import testing
from . import _funcs


@testing.parametrized(
    shifted_elliptic=(_funcs.ShiftedElliptic, ([12, 24],), [0, 10], 235571325.2279835),  # f1
    shifted_elliptic_0=(_funcs.ShiftedElliptic, ([12, 24],), [12, 24], 0),  # f1
    shifted_rastrigin=(_funcs.ShiftedRastrigin, ([12, 24],), [10, 20], 179.03902),  # f2
    shifted_rastrigin_0=(_funcs.ShiftedRastrigin, ([12, 24],), [12, 24], 0),  # f2
    shifted_ackley=(_funcs.ShiftedAckley, ([12, 24],), [10, 20], 18.1190856),  # f3
    shifted_ackley_0=(_funcs.ShiftedAckley, ([12, 24],), [12, 24], 0),  # f3
    shifted_rosenbrock=(_funcs.ShiftedRosenbrock, ([12, 24],), [10, 20], 6409.0),  # f12
    shifted_rosenbrock_0=(_funcs.ShiftedRosenbrock, ([12, 24],), [13, 25], 0),  # f12, shift + 1
    shifted_schwefel=(_funcs.ShiftedSchwefel, ([12, 24],), [10, 20], 40.48019),  # f15
    shifted_schwefel_0=(_funcs.ShiftedSchwefel, ([12, 24],), [12, 24], 0),  # f15
)
def test_testbed_functions(
    cls: Callable[..., Callable[[np.ndarray], float]], params: List[Any], data: List[float], expected: float
) -> None:
    np.random.seed(12)
    func = cls(*[np.array(p) for p in params])
    value = func(np.array(data))
    np.testing.assert_almost_equal(value, expected, decimal=5)


def test_read_data() -> None:
    data = _funcs.read_data("F4")
    testing.assert_set_equal(data.keys(), ["p", "s", "w", "xopt", "R50", "R25", "R100"])
    assert data["R50"].shape == (50, 50)
    np.testing.assert_almost_equal(
        data["R100"][:2, :2],
        [[-0.04915187244640615, 0.09617611414762256], [0.04609873606566875, 0.07060894383640659]],
    )


@testing.parametrized(  # manually verified against approximate Octave and CPP values on vector of zeros
    f1=(1, 1000, 209833896353.34344),
    f2=(2, 1000, 47620.31161660616),
    f3=(3, 1000, 21.72900253494246),  # Octave version needs uncommenting Tosz, Tasy, Lambda
    f4=(4, 1000, 107955147656065.94),
    f5=(5, 1000, 48419148.33292),
    f6=(6, 1000, 1077732.46530),  # Octave version needs uncommenting Tosz, Tasy, Lambda
    f7=(7, 1000, 993826981321072.6),  # same as implementations, which do not add transf on the side loss
    f8=(8, 1000, 5.722271501878064e18),
    f9=(9, 1000, 6001603202.501935),
    f10=(10, 1000, 98115481.6486656),  # Octave version uncommenting Tosz, Tasy, Lambda
    f11=(11, 1000, 1.0448520164721186e17),
    f12=(12, 1000, 1711354236949.7197),
    f13=(13, 905, 8.273800489859654e16),  # fails with octave
    f14=(14, 905, 4.4079796812096026e18),  # unspecified optimum
    f15=(15, 1000, 2393892336615502.5),
)
def test_functions_zeros(number: int, dimension: int, expected: float) -> None:
    func = _funcs.make_function(number)
    assert func.dimension == dimension
    value = func(np.array([0] * func.dimension))
    np.testing.assert_approx_equal(value, expected, significant=12)
    if number != 14:
        assert func.optimum is not None
        np.testing.assert_almost_equal(func(func.optimum), 0)


@testing.parametrized(  # verification against exact CPP outputs on vector of ones
    f1=(1, 15, 2.09946678145388153076e11),
    f2=(2, 15, 7.00495371043751511024e04),
    f3=(3, 15, 2.17108415925776405686e01),
    f4=(4, 15, 1.07162206769653859375e14),
    f5=(5, 15, 5.87148888268808051944e07),
    f6=(6, 15, 1.07977197180324327201e06),
    f7=(7, 14, 9.29113705518042875000e14),
    f8=(8, 15, 5.60788325599985049600e18),
    f9=(9, 15, 9.44072284529276657104e09),
    f10=(10, 13, 9.78947871248564124107e07),
    f11=(11, 15, 1.01442464039521104000e17),
    f12=(12, 15, 1.71217696529957031250e12),
    f13=(13, 15, 9.69220815693190400000e16),
    f14=(14, 14, 4.37551256977279180800e18),
    f15=(15, 15, 2.75152052424948050000e15),
)
def test_functions_ones(number: int, significant: int, expected: float) -> None:
    func = _funcs.make_function(number)
    value = func(np.array([1] * func.dimension))
    np.testing.assert_approx_equal(value, expected, significant=min(14, significant))
