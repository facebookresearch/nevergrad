# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from . import cycling


def test_cycling() -> None:
    func = cycling.Cycling(30)
    x = 0 * np.random.rand(func.dimension)
    assert False, f"{func(x)}"
