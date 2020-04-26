# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import numpy as np
import nevergrad as ng
from . import core


def test_images() -> None:
    func = core.Images()
    x = 7 * np.random.normal(size=TODO)
    data = func.parametrization.spawn_child().set_standardized_data(x).args[0]
    value = func(data)  # should not touch boundaries, so value should be < np.inf
    assert value < np.inf
    data = func.parametrization.spawn_child().set_standardized_data(np.arange(8)).args[0]
    for f in [func, func.evaluation_function]:
        np.testing.assert_almost_equal(f(data), 13.1007174)  # type: ignore
