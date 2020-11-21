# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym
import numpy as np


class GenericMujocoEnv:
    """This class evaluates policy of OpenAI Gym environment.

    Parameters
    -----------
    env_name: str
        Gym environment name
    state_mean: list
        Average state values of multiple independent runs.
    state_std: list
        Standard deviation of state values of multiple independent runs.
    num_rollouts: int
        number of independent runs.
    random_state: int or None
        random state for reproducibility in Gym environment.
    """

    def __init__(self, env_name, num_rollouts,
                 random_state, online_stats, state_mean=None, state_std=None):
        self.online_stats = online_stats
        self.s_n = 0.
        self.n_obs = 0

        if self.online_stats:
            self.mean = 0.
            self.std = 0.
        else:
            self.mean = np.array(state_mean)
            self.std = np.array(state_std)

        self.env = gym.make(env_name)
        self.num_rollouts = num_rollouts
        self.env.seed(random_state)

    def __call__(self, x):
        """Compute average cummulative reward of a given policy.
        """
        returns = []
        for _ in range(self.num_rollouts):
            obs = self.env.reset()
            if self.online_stats:
                self.update_stats(obs)
            done = False
            totalr = 0.
            while not done:
                action = np.dot(x, (obs - self.mean) / (self.std + 1e-9))
                obs, r, done, _ = self.env.step(action)
                if self.online_stats:
                    self.update_stats(obs)
                totalr += r
            returns.append(totalr)

        return -np.mean(returns)

    def update_stats(self, obs):
        self.n_obs += 1
        e = obs - self.mean
        self.mean += e / self.n_obs
        self.s_n += e * (obs - self.mean)
        self.std = self.s_n / max(self.n_obs - 1, 1)
