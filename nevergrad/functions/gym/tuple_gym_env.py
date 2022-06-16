# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym
import numpy as np
from gym.spaces import Box, Tuple, MultiDiscrete


class TupleActionSpace(gym.Env):
    """
    Environment with tuple of discrete spaces at each time step.
    """

    def __init__(self):

        super().__init__()
        self.current_step = 0
        self.num_cells = 71
        self.action_space = Tuple((MultiDiscrete(self.num_cells * [2]), MultiDiscrete(self.num_cells * [6])))
        self.horizon = 168
        self._reward = 0.0

        self.observation_space = Box(
            low=np.array([0, 0, 0]),
            high=np.array([1, 1, 1]),
            dtype=np.float16,
        )

    def _next_observation(self):

        return np.asarray((0.5, 0.5, 0.5), dtype="float16")

    def reward(
        self,
    ) -> float:
        return float(self._reward)

    def _take_action(self, action):
        assert len(action) == 2
        assert len(action[0]) == self.num_cells
        for u in action[1]:
            assert int(u) == u
        self._reward = np.sum(action[1]) - np.sum(action[0])
        return self._reward

    def reset(self):
        return self._next_observation()

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        obs = self._next_observation()
        self.current_step += 1
        return obs, self.reward(), self.current_step == self.horizon, {}
