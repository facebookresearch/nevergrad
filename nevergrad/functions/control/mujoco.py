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
    activation: str:
        activation function
    layer_rescaling_coef: float
        Scaling coefficient of output layers
    random_state: int or None
        random state for reproducibility in Gym environment.
    """

    def __init__(
        self,
        env_name,
        state_mean,
        state_std,
        num_rollouts,
        activation,
        layer_rescaling_coef,
        noise_level,
        random_state,
    ):
        self.mean = state_mean
        self.std = state_std
        self.env = gym.make(env_name)
        self.num_rollouts = num_rollouts
        self.random_state = random_state
        self.activation = activation
        self.layer_rescaling_coef = layer_rescaling_coef
        self.noise_level = noise_level

    def _activation(self, x):
        if self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "sigmoid":
            return 1.0 / (1 + np.exp(-x))
        else:
            raise NotImplementedError(r"Activation {self.activation} not implemented.")

    def __call__(self, layers):
        """Compute loss (average cumulative negative reward) of a given policy."""
        returns = []
        for _ in range(self.num_rollouts):
            obs = self.env.reset()
            done = False
            totalr = 0.0
            while not done:
                action = (
                    np.matmul(obs, layers[0])
                    if (self.mean is None)
                    else (np.matmul((obs - self.mean) / self.std, layers[0]))
                )
                action = action * self.layer_rescaling_coef[0]
                for x, r_coef in zip(layers[1:], self.layer_rescaling_coef[1:]):
                    action = np.matmul(self._activation(action) + 1.0e-3, x) * r_coef
                if self.noise_level > 0.0:
                    action += action * self.noise_level * self.random_state.normal(size=action.shape)
                obs, r, done, _ = self.env.step(action)
                totalr += r
            returns.append(totalr)

        return -np.mean(returns)
