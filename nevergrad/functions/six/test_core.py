# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from . import core


def test_six() -> None:
    func = core.SixGame()
    x = func.parametrization.sample()
    value = func(x.value)
    np.testing.assert_almost_equal(value, 199)
