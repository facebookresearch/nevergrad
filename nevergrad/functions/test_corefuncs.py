# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Any, List, Optional
import numpy as np
from ..common import testing
from . import corefuncs


@testing.parametrized(**{name: (name, func) for name, func in corefuncs.registry.items()})
def testcorefuncs_function(name: str, func: Callable[..., Any]) -> None:
    x = np.random.normal(0, 1, 100)
    outputs = []
    for _ in range(2):
        np.random.seed(12)
        outputs.append(func(x))
    np.testing.assert_equal(outputs[0], outputs[1], f'Function {name} is not deterministic')


@testing.parametrized(
    expe1=([6, 4, 2, 1, 9], 4, 5, 3),  # jump was assumed correct (verify?)
    expe2=([6, 6, 7, 1, 9], 4, 5, 3),
)
def test_base_functions(x: List[int], onemax_expected: float, leadingones_expected: float, jump_expected: float) -> None:
    np.testing.assert_equal(corefuncs._onemax(x), onemax_expected, err_msg="Wrong output for onemax")
    np.testing.assert_equal(corefuncs._leadingones(x), leadingones_expected, err_msg="Wrong output for leadingones")
    np.testing.assert_equal(corefuncs._jump(x), jump_expected, err_msg="Wrong output for jump")


def test_genzcornerpeak_inf() -> None:
    y = [-np.inf, -np.inf]
    output = corefuncs.genzcornerpeak(y)
    np.testing.assert_equal(output, np.inf)
    output = corefuncs.minusgenzcornerpeak(y)
    np.testing.assert_equal(output, -np.inf)


@testing.parametrized(
    cigar=(corefuncs.cigar, 8604785.43824, None),
    hm=(corefuncs.hm, 15.85037, None),
    griewank=(corefuncs.griewank, 0.70028, None),
    sphere=(corefuncs.sphere, 9.043029, None),
    sphere1=(corefuncs.sphere1, 10.329029, None),
    sphere2=(corefuncs.sphere2, 35.615029, None),
    sphere4=(corefuncs.sphere4, 158.187029, None),
    sphere_m=(corefuncs.sphere, 30, [1, 2, 3, 4]),
    elliptic=(corefuncs.ellipsoid, 262389.541996, None),
    altelliptic=(corefuncs.altellipsoid, 609313.71475, None),
    rosenbrock=(corefuncs.rosenbrock, 1967.57859, None),
    rosenbrock_m=(corefuncs.rosenbrock, 2705, [1, 2, 3, 4]),
)
def test_core_function(func: Callable[[np.ndarray], float], expected: float, data: Optional[List[float]]) -> None:
    if data is None:
        data = [0.662, -0.217, -0.968, 1.867, 0.101, 0.575, 0.199, 1.576, 1.006, 0.182, -0.092, 0.466]
    value = func(np.array(data))
    np.testing.assert_almost_equal(value, expected, decimal=5)
