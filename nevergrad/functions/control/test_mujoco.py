# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import numpy as np
from .core import BaseFunction

class MountainCarContinuous(BaseFunction):
    env_name = "MountainCarContinuous-v0"
    policy_dim= (2, 1)
    state_mean = [0, 0]
    state_std = [1, 1]

def test_gym() -> None:
    func = MountainCarContinuous(num_rollouts=5)

    x = func.parametrization.value
    np.testing.assert_array_equal(x[0][0], [[0.], [0.]])
    value = func(x)
    np.testing.assert_almost_equal(value, 0.)

    x = func.parametrization.sample().value
    assert not np.array_equal(x[0][0], [[0.], [0.]])
    value = func(func.parametrization.sample().value)
    assert value != 0

def test_all_mujoco_envs() -> None:
    pytest.importorskip('mujoco_py')
    core = pytest.importorskip('nevergrad.functions.control.core')

    for module in ["Ant", "Swimmer", "HalfCheetah", "Hopper", "Walker2d", "Humanoid", "NeuroAnt", "NeuroSwimmer", "NeuroHalfCheetah", "NeuroHopper", "NeuroWalker2d", "NeuroHumanoid"]:
        func = getattr(core, module)(num_rollouts=1)
        for _ in range(10):
            assert isinstance(func(func.parametrization.sample().value), float)