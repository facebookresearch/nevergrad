# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from . import core


def test_mixsimulator() -> None:
    np.random.seed(17)
    func = core.OptimizeFish()
    x = np.random.rand(func.dimension)
    value = func(x)  # should not touch boundaries, so value should be < np.inf
    np.testing.assert_almost_equal(value, -0.985, decimal=2)
