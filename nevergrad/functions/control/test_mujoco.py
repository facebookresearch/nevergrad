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


@pytest.mark.parametrize("intermediate_layer_dim", [None, (4,), (3, 3)])
@pytest.mark.parametrize("noise_level", [0.0, 0.9])
@pytest.mark.parametrize("deterministic_sim", [0.0, 0.9])
@pytest.mark.parametrize("states_normalization", [True, False])
def test_gym(intermediate_layer_dim, noise_level, deterministic_sim, states_normalization) -> None:
    func = MountainCarContinuous(
        num_rollouts=2,
        intermediate_layer_dim=intermediate_layer_dim,
        noise_level=noise_level,
        deterministic_sim=deterministic_sim,
        states_normalization=states_normalization,
        random_state=42,
    )

    x = func.parametrization.value
    value = func(x)
    np.testing.assert_almost_equal(value, 0.0)
    x = func.parametrization.sample().value
    assert func(x) != 0
    if noise_level == 0.0 and deterministic_sim:
        assert func(x) == func(x)
    else:
        assert func(x) != func(x)


@pytest.mark.parametrize("module", ["Ant", "Swimmer", "HalfCheetah", "Hopper", "Walker2d", "Humanoid"])
@pytest.mark.parametrize("intermediate_layer_dim", [None, (50,)])
@pytest.mark.parametrize("noise_level", [0.0, 0.9])
@pytest.mark.parametrize("deterministic_sim", [0.0, 0.9])
@pytest.mark.parametrize("states_normalization", [True, False])
def test_all_mujoco_envs(
    module, intermediate_layer_dim, noise_level, deterministic_sim, states_normalization
) -> None:
    pytest.importorskip("mujoco_py")
    core = pytest.importorskip("nevergrad.functions.control.core")
    func = getattr(core, module)(
        num_rollouts=1,
        intermediate_layer_dim=intermediate_layer_dim,
        noise_level=noise_level,
        deterministic_sim=deterministic_sim,
        states_normalization=states_normalization,
        random_state=42,
    )
    for _ in range(3):
        assert isinstance(func(func.parametrization.sample().value), float)
