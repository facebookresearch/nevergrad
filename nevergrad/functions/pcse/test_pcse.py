# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from . import pcse


def test_pcse() -> None:
    func = pcse.CropSimulator()
    x = 0 * np.random.rand(func.dimension)
    value = func(x)
    value2 = func(x)
    assert value > -1000.0  # type: ignore
    assert value < 1000.0  # type: ignore
    np.testing.assert_almost_equal(value, value2)
