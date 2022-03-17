# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from . import irrigation


def test_irrigation() -> None:
    func = irrigation.Irrigation(3)
    x = np.random.rand(func.dimension)
    value = func(x)
    value2 = func(x)
    x = np.random.rand(func.dimension)
    value3 = func(x)
    assert value <= 0.0  # type: ignore
    assert value3 != value  # this should not be flat.
    np.testing.assert_almost_equal(value, value2)
