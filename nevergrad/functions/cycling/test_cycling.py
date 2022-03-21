# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from . import cycling


def test_cycling() -> None:
    func = cycling.Cycling(30)
    x = 0 * np.random.rand(func.dimension)
    assert func(x)
    np.testing.assert_almost_equal(func(x), 282.8, decimal=1)
    for index in [31, 22, 23, 45, 61]:
        func2 = cycling.Cycling(index)
        strategy = func2.parametrization.sample().value
        func2(strategy)
