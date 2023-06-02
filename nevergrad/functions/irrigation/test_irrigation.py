# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
from . import irrigation


@pytest.mark.parametrize("i", list(range(7)))
def test_irrigation(i: int) -> None:
    func = irrigation.Irrigation(i)
    x = np.random.rand(func.dimension)
    value = func(x)
    value2 = func(x)
    value3 = func(np.random.rand(func.dimension))
    value4 = func(np.random.rand(func.dimension))
    value5 = func(np.random.rand(func.dimension))
    value6 = func(np.random.rand(func.dimension))
    value7 = func(np.random.rand(func.dimension))
    value8 = func(np.random.rand(func.dimension))
    assert value <= 0.0  # type: ignore
    v = [value, value3, value4, value5, value6, value7, value8]
    assert min(v) != max(v)  # this should not be flat.
    np.testing.assert_almost_equal(value, value2)
