# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import typing as tp
from .mujoco import GenericMujocoEnv
from ..base import ExperimentFunction
from nevergrad.parametrization import parameter as p


def test_gym() -> None:
    class MountainCarContinuous(ExperimentFunction):
        state_mean = [0, 0]
        state_std = [1, 1]

        def __init__(self, num_rollouts: int, random_state: tp.Optional[int] = None) -> None:
            super().__init__(self._simulate_env, p.Array(shape=(2,)))
            self.num_rollouts = num_rollouts
            self.random_state = random_state
            self.register_initialization(num_rollouts=num_rollouts, random_state=random_state)

        def _simulate_env(self, x: np.ndarray) -> float:
            env = GenericMujocoEnv(env_name="MountainCarContinuous-v0",
                                   state_mean=self.state_mean,
                                   state_std=self.state_std,
                                   policy_dim=(1, 2),
                                   num_rollouts=self.num_rollouts,
                                   random_state=self.random_state)
            return env(x)

    func = MountainCarContinuous(num_rollouts=5)
    value = func(np.zeros(func.dimension))
    np.testing.assert_almost_equal(value, 0.)
    value = func(np.random.rand(func.dimension))
    assert value != 0
