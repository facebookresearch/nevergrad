# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import os
import gym

if os.name != "nt":
    import gym_anm  # pylint: disable=unused-import
from nevergrad.parametrization import parameter
from ..base import ExperimentFunction


# Method for building a new list, for a future version of gym:
#
# GYM_ENV_NAMES = ["gym_anm:ANM6Easy-v0"]
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
#         GYM_ENV_NAMES.append(e.id)
#     except:
#         pass

GYM_ENV_NAMES = [
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
    "Taxi-v3",
    "CubeCrash-v0",
    "CubeCrashSparse-v0",
    "CubeCrashScreenBecomesBlack-v0",
    "MemorizeDigits-v0",
]


class GymMulti(ExperimentFunction):
    def __init__(
        self, name: str = "gym_anm:ANM6Easy-v0", control: str = "conformant", neural_factor: int = 2
    ) -> None:
        env = gym.make(name)
        self.name = name + "__" + control + "__" + str(neural_factor)
        self.num_time_steps = 100
        self.neural_factor = neural_factor
        o = env.reset()
        if "int" in str(type(env.action_space.sample())):  # Discrete action space
            output_dim = env.action_space.n
            output_shape = (output_dim,)
            discrete = True
        else:  # Continuous action space
            output_shape = tuple(np.asarray(env.action_space.sample()).shape)  # type: ignore
            discrete = False
            output_dim = np.prod(output_shape)
        if "int" in str(type(o)):
            input_dim = env.observation_space.n
            self.discrete_input = True
        else:
            input_dim = np.prod(np.asarray(o).shape)
            self.discrete_input = False
        self.action_type = type(env.action_space.sample())
        self.output_shape = output_shape
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = neural_factor * input_dim
        assert control in [
            "conformant",
            "linear",
            "neural",
            "noisy_neural",
            "noisy_scrambled_neural",
            "scrambled_neural",
        ], f"{control} not known as a form of control"
        self.control = control
        neural_size = (output_dim * self.num_neurons + self.num_neurons * (input_dim + 1),)
        shape = {
            "conformant": (self.num_time_steps,) + output_shape,
            "linear": (input_dim + 1, output_dim),
            "neural": neural_size,
            "noisy_neural": neural_size,
            "noisy_scrambled_neural": neural_size,
            "scrambled_neural": neural_size,
        }[control]
        shape = tuple(int(s) for s in shape)
        self.policy_shape = shape
        parametrization = parameter.Array(shape=shape)
        super().__init__(self.gym_multi_function, parametrization=parametrization)
        self.env = env
        self.discrete = discrete

    def env_names(self):
        return GYM_ENV_NAMES

    def discretize(self, a):
        probabilities = np.exp(a - max(a))
        probabilities = probabilities / sum(probabilities)
        return int(np.random.multinomial(1, probabilities)[0])

    def neural(self, x: np.ndarray, o: np.ndarray):
        x = x.reshape(self.policy_shape)
        if self.control == "linear":
            output = np.matmul(o.ravel(), x[1:, :])
            output += x[0]
            return output.reshape(self.output_shape)
        first_size = self.num_neurons * (self.input_dim + 1)
        first_matrix = x[:first_size].reshape(self.input_dim + 1, self.num_neurons)
        second_matrix = x[first_size:].reshape(self.num_neurons, self.output_dim)
        return np.matmul(
            np.tanh(np.matmul(o.ravel(), first_matrix[1:]) + first_matrix[0]), second_matrix
        ).reshape(self.output_shape)

    def gym_multi_function(self, x: np.ndarray):
        reward = 0.0
        num_simulations = 7 if self.control != "conformant" else 1
        for _ in range(num_simulations):
            reward += self.gym_simulate(x)
        return reward / num_simulations

    def gym_simulate(self, x: np.ndarray, seed: int = 0):
        assert seed == 0 or self.control != "conformant"
        env = self.env
        env.seed(seed=seed)
        o = env.reset()
        control = self.control
        if control == "conformant":
            return self.gym_conformant(x)
        if "scrambled" in control:
            np.random.RandomState(1234).shuffle(x)
        if "noisy" in control:
            x = x + 0.01 * np.random.RandomState(1234).normal(size=x.shape)
        reward = 0.0
        for i in range(self.num_time_steps):
            if self.discrete_input:
                obs = np.zeros(shape=self.input_dim)
                obs[o] = 1
                o = obs
            o = np.asarray(o)
            a = self.neural(x, o)
            if self.discrete:
                a = self.discretize(a)
            try:
                a = self.action_type(a)
                assert type(a) == self.action_type
                o, r, done, _ = env.step(a)  # Outputs = observation, reward, done, info.
            except AssertionError:  # Illegal action.
                return 1e20 / (1.0 + i)  # We encourage late failures rather than early failures.
            reward += r
            if done:
                break
        return reward

    def gym_conformant(self, x: np.ndarray):
        reward = 0.0
        for i, a in enumerate(10.0 * x):
            if type(a) == np.float64:
                a = np.asarray((a,))
                if self.discrete:
                    a = self.discretize(a)
            try:
                a = self.action_type(a)
                assert type(a) == self.action_type
                _, r, done, _ = self.env.step(a)  # Outputs = observation, reward, done, info.
            except AssertionError:  # Illegal action.
                return 1e20 / (1.0 + i)  # We encourage late failures rather than early failures.
            reward += r
            if done:
                break
        # env.render()  if you want to display.
        return -reward
