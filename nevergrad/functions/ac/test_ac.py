# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from . import ac


def test_ac() -> None:
    func = ac.NgAquacrop(4, 12.0)
    x = 50.0 * np.random.rand(func.dimension)
    value = func(x)
    value2 = func(x)
    x = 50.0 * np.random.rand(func.dimension)
    value3 = func(x)
    np.testing.assert_almost_equal(value, value2)
    assert value != value3
