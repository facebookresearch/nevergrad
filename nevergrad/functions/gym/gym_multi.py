# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import gym
import os

if os.name != "nt":
    import gym_anm  # type: ignore  # pylint: disable=unused-import
from nevergrad.parametrization import parameter
from ..base import ExperimentFunction

gym_env_names = ["gym_anm:ANM6Easy-v0"] 

for e in gym.envs.registry.all():
    try:
        gym.make(e.id)
        gym_env_names.append(e.id)
    except:
        pass


#+ [e.id for e in gym.envs.registry.all() if not any(x in e.id for x in ["Lunar", "BipedalWalker", "CarRacing", "Reacher", "Pusher", "Thrower", "Striker", "InvertedPendulum", "InvertedDoublePendulum", "HalfCheetah", "Swimmer", "Hopper", "Walker2d", "Ant", "Humanoid", "Fetch", "HandReach", "HandManipulate"])]

#    ["MountainCarContinuous-v0", "CartPole-v0", "FrozenLake8x8-v0", "Breakout-v0"]
#    from gym import envs
#    all_envs = envs.registry.all()
#    env_ids = [env_spec.id for env_spec in all_envs]
#    print(env_ids)

class GymMulti(ExperimentFunction):
    def __init__(self, name: str = "gym_anm:ANM6Easy-v0") -> None:
        env = gym.make(name)
        if "int" in str(type(env.action_space.sample())):  # Discrete action space
            dimension = (env.action_space.n,)
            discrete = True
        else:  # Continuous action space
            dimension = tuple(np.asarray(env.action_space.sample()).shape)
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
