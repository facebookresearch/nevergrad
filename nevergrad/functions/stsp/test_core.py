# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
from . import core


@pytest.mark.parametrize("complex_tsp", [True, False])
def test_stsp(complex_tsp: bool) -> None:
    func = core.STSP(complex_tsp=complex_tsp)
    x = 7 * np.random.rand(func.dimension)
    value = func(x)  # should not touch boundaries, so value should be < np.inf
    assert isinstance(value, float)
    assert value < np.inf
