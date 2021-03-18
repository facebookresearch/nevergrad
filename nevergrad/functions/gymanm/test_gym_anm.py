# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
from nevergrad.common import testing
from . import gym_anm


def test_gym_anm() -> None:
    if os.name != "nt":
        func = gym_anm.GymAnm()
        x = np.zeros(func.dimension)
        value = func(x)
        assert value == 1e20
