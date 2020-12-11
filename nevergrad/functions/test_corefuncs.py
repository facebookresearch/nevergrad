# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import numpy as np
from nevergrad.common import testing
from . import corefuncs


@testing.parametrized(**{name: (name, func) for name, func in corefuncs.registry.items()})
def testcorefuncs_function(name: str, func: tp.Callable[..., tp.Any]) -> None:
    x = np.random.normal(0, 1, 100)
    outputs = []
    for _ in range(2):
        np.random.seed(12)
        outputs.append(func(x))
    np.testing.assert_equal(outputs[0], outputs[1], f"Function {name} is not deterministic")


@testing.parametrized(
    expe1=([6, 4, 2, 1, 9], 4, 5, 3),
    expe2=([6, 6, 7, 1, 9], 4, 5, 3),
    expe3=([1, 1, 7, 1, 9], 3, 5, 2),
    expe4=([0, 0, 7, 1, 9], 3, 4, 2),
    expe5=([0, 1, 1, 1], 1, 2, 0),
    expe6=([1, 1, 1, 1], 2, 4, 1),
    expe7=([1, 0, 0, 0], 3, 4, 2),
    expe_0lead=([0, 1, 0, 1], 0, 0, -1),
)  # jump was assumed correct (verify?)
def test_base_functions(
    x: tp.List[float], onemax_expected: float, leadingones_expected: float, jump_expected: float
) -> None:
    for name, expected in [
        ("onemax", onemax_expected),
        ("leadingones", leadingones_expected),
        ("jump", jump_expected),
    ]:
        func = corefuncs.DiscreteFunction(name)
        np.testing.assert_equal(func(x), expected, err_msg=f"Wrong output for {name}")


def test_genzcornerpeak_inf() -> None:
    y = np.array([-np.inf, -np.inf])
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
    ackley=(corefuncs.ackley, 4.7298473, None),
    schwefel=(corefuncs.schwefel_1_2, 128.709765, None),
    ackley_m=(corefuncs.ackley, 2.254444, [0.1, 0.2, 0.3]),
    schwefel_m=(corefuncs.schwefel_1_2, 146, [1, 2, 3, 4]),
    genzgaussianpeakintegral=(corefuncs.genzgaussianpeakintegral, 0.10427, None),
    minusgenzgaussianpeakintegral=(corefuncs.minusgenzgaussianpeakintegral, -0.10427, None),
    linear=(corefuncs.linear, 0.57969, None),
)
def test_core_function_values(
    func: tp.Callable[[np.ndarray], float], expected: float, data: tp.Optional[tp.List[float]]
) -> None:
    if data is None:
        data = [0.662, -0.217, -0.968, 1.867, 0.101, 0.575, 0.199, 1.576, 1.006, 0.182, -0.092, 0.466]
    value = func(np.array(data))
    np.testing.assert_almost_equal(value, expected, decimal=5)


def test_styblinksitang() -> None:
    np.random.seed(12)
    data = [0.662, -0.217, -0.968, 1.867, 0.101, 0.575, 0.199, 1.576, 1.006, 0.182, -0.092, 0.466]
    value = corefuncs._styblinksitang(np.array(data), noise=0.1)
    np.testing.assert_almost_equal(value, 421.374940, decimal=5)
