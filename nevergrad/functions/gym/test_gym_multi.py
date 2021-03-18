# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
from . import gym_multi


def test_gym_multi() -> None:
    if os.name != "nt":
        func = gym_multi.GymMulti()
        x = np.zeros(func.dimension)
        value = func(x)
        assert value == 1e20
    for name in gym_multi.gym_env_names:
        func = gym_multi.GymMulti(name)
        x = np.zeros(func.dimension)
        value = func(x)
        print(f"{name} -> {value}")
    assert len(gym_multi.gym_env_names) == 29  # For the moment, this includes 29 environments.
