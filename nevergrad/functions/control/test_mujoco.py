# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from .core import BaseFunction

class MountainCarContinuous(BaseFunction):
    env_name = "MountainCarContinuous-v0"
    policy_dim= (1, 2)
    state_mean = [0, 0]
    state_std = [1, 1]

def test_gym() -> None:
    func = MountainCarContinuous(num_rollouts=5)

    x = func.parametrization.value
    np.testing.assert_array_equal(x, [[0., 0.]])
    value = func(x)
    np.testing.assert_almost_equal(value, 0.)

    x = func.parametrization.sample().value
    assert not np.array_equal(x, [[0., 0.]])
    value = func(func.parametrization.sample().value)
    assert value != 0
