# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from unittest.mock import patch
import numpy as np
from . import core


def test_ttp() -> None:
    np.random.seed(17)

    func = core.TTPInstance()
    tsp_tour = np.random.random_sample((19,))
    packing_plan = np.random.random_sample((95,))
    packing_plan = packing_plan > 0.5
    packing_plan = packing_plan.astype(int)
    value = func.function(tsp_tour, packing_plan)
    np.testing.assert_almost_equal(value, 21071817.386196714)
