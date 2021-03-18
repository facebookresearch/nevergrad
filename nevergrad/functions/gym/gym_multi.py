# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import os
import gym

if os.name != "nt":
    import gym_anm  # type: ignore  # pylint: disable=unused-import
from nevergrad.parametrization import parameter
from ..base import ExperimentFunction


# Method for building a new list, for a future version of gym:
#
# gym_env_names = ["gym_anm:ANM6Easy-v0"]
#
# for e in gym.envs.registry.all():
#     try:
#         assert "Kelly" not in e.id
#         env = gym.make(e.id)
#         a1 = env.action_space.sample()
#         a2 = env.action_space.sample()
#         a3 = env.action_space.sample()
#         a1 = a1 + a2 + a3
#         if hasattr(a1, "size"):
#             assert a1.size() < 15000
#         gym_env_names.append(e.id)
#     except:
#         pass

gym_env_names = [
    "gym_anm:ANM6Easy-v0",
    "Copy-v0",
    "RepeatCopy-v0",
    "ReversedAddition-v0",
    "ReversedAddition3-v0",
    "DuplicatedInput-v0",
    "Reverse-v0",
    "CartPole-v0",
    "CartPole-v1",
    "MountainCar-v0",
    "Acrobot-v1",
    "Blackjack-v0",
    "FrozenLake-v0",
    "FrozenLake8x8-v0",
    "CliffWalking-v0",
    "NChain-v0",
    "Roulette-v0",
    "Taxi-v2",
    "CubeCrash-v0",
    "CubeCrashSparse-v0",
    "CubeCrashScreenBecomesBlack-v0",
    "MemorizeDigits-v0",
]


class GymMulti(ExperimentFunction):
    def __init__(self, name: str = "gym_anm:ANM6Easy-v0") -> None:
        env = gym.make(name)
        if "int" in str(type(env.action_space.sample())):  # Discrete action space
            dimension = (env.action_space.n,)
            discrete = True
        else:  # Continuous action space
            dimension = tuple(np.asarray(env.action_space.sample()).shape)  # type: ignore
            discrete = False
        shape = (100,) + dimension
        super().__init__(self.gym_multi_function, parametrization=parameter.Array(shape=shape))
        self.env = env
        self.discrete = discrete

    def env_names(self):
        return gym_env_names

    def gym_multi_function(self, x: np.ndarray):

        env = self.env
        env.seed(0)
        _ = env.reset()  # output value = "o"

        reward = 0.0
        for i, val in enumerate(x):
            a = 10.0 * val
            if type(a) == np.float64:
                a = np.asarray((a,))
            if self.discrete:
                probabilities = np.exp(a - max(a))
                probabilities = probabilities / sum(probabilities)
                a = np.random.multinomial(1, probabilities)[0]
            try:
                _, r, done, _ = env.step(a)  # Outputs = observation, reward, done, info.
            except AssertionError:  # Illegal action.
                return 1e20 / (1.0 + i)  # We encourage late failures rather than early failures.
            reward += r
            if done:
                break
        # env.render()  if you want to display.
        return -reward
