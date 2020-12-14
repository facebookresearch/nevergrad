# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import numpy as np
from .core import BaseFunction


class MountainCarContinuous(BaseFunction):
    env_name = "MountainCarContinuous-v0"
    policy_dim = (2, 1)
    state_mean = [0, 0]
    state_std = [1, 1]


def test_gym() -> None:
    for intermediate_layer_dim in [None, (10,), (5, 5)]:
        for noise_level in [0.0, 0.9]:
            for deterministic_sim in [True, False]:
                for states_normalization in [True, False]:
                    func = MountainCarContinuous(num_rollouts=5, intermediate_layer_dim=intermediate_layer_dim,
                                                 noise_level=noise_level, deterministic_sim=deterministic_sim,
                                                 states_normalization=states_normalization, random_state=42)

                    x = func.parametrization.value
                    value = func(x)
                    np.testing.assert_almost_equal(value, 0.)
                    x = func.parametrization.sample().value
                    assert func(x) != 0
                    if noise_level == 0.0 and deterministic_sim:
                        assert func(x) == func(x)
                    else:
                        assert func(x) != func(x)


def test_all_mujoco_envs() -> None:
    pytest.importorskip('mujoco_py')
    core = pytest.importorskip('nevergrad.functions.control.core')

    for module in ["Ant", "Swimmer", "HalfCheetah", "Hopper", "Walker2d", "Humanoid"]:
        for intermediate_layer_dim in [None, (50,)]:
            for noise_level in [0.0, 0.9]:
                for deterministic_sim in [True, False]:
                    for states_normalization in [True, False]:
                        func = getattr(core, module)(num_rollouts=1, intermediate_layer_dim=intermediate_layer_dim,
                                                 noise_level=noise_level, deterministic_sim=deterministic_sim,
                                                 states_normalization=states_normalization, random_state=42)
                        for _ in range(3):
                            assert isinstance(func(func.parametrization.sample().value), float)
