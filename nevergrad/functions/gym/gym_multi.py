# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import typing as tp
import gym
import compiler_gym
import nevergrad as ng

if os.name != "nt":
    import gym_anm  # pylint: disable=unused-import
from nevergrad.parametrization import parameter
from ..base import ExperimentFunction


## Method for building a new list, for a future version of gym:

GYM_ENV_NAMES = []

for e in gym.envs.registry.all():
    try:
        assert "Kelly" not in str(e.id)
        env = gym.make(e.id)
        env.reset()
        env.step(env.action_space.sample())
        a1 = np.asarray(env.action_space.sample())
        a2 = np.asarray(env.action_space.sample())
        a3 = np.asarray(env.action_space.sample())
        a1 = a1 + a2 + a3
        if hasattr(a1, "size"):
            try:
                assert a1.size < 15000  # type: ignore
            except:
                assert a1.size() < 15000  # type: ignore
        GYM_ENV_NAMES.append(e.id)
        # print(f"adding {e.id}, {len(GYM_ENV_NAMES)} environments...")
    except:
        pass

GUARANTEED_GYM_ENV_NAMES = [
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


# We do not use "conformant" which is not consistent with the rest.
CONTROLLERS = [
    "linear",  # Simple linear controller.
    "neural",  # Simple neural controller.
    "deep_neural",  # Deeper neural controller.
    "semideep_neural",  # Deep, but not very deep.
    "structured_neural",  # Structured optimization of a neural net.
    "memory_neural",  # Uses a memory (i.e. recurrent net).
    "deep_memory_neural",
    "stackingmemory_neural",  # Uses a memory and stacks a heuristic and the memory as inputs.
    "deep_stackingmemory_neural",
    "semideep_stackingmemory_neural",
    "extrapolatestackingmemory_neural",  # Same as stackingmemory_neural + suffix-based extrapolation.
    "deep_extrapolatestackingmemory_neural",
    "semideep_extrapolatestackingmemory_neural",
    "semideep_memory_neural",
    "multi_neural",  # One neural net per time step.
    "noisy_neural",  # Do not start at 0 but at a random point.
    "scrambled_neural",  # Why not perturbating the order of variables ?
    "noisy_scrambled_neural",
    "stochastic_conformant",  # Conformant planning, but still not deterministic.
]


NO_LENGTH = ["ANM", "Blackjack", "CliffWalking", "Cube", "Memorize", "ompiler"]


# Environment used for CompilerGym.
class SmallActionSpaceLlvmEnv(gym.ActionWrapper):
    """A wrapper for the LLVM compiler environment that exposes a tiny subset of
    the full discrete action space (the subset was hand pruned to contain a mix
    of "good" and "bad" actions).
    """

    action_space_subset = [
        "-adce",
        "-break-crit-edges",
        "-constmerge",
        "-correlated-propagation",
        "-deadargelim",
        "-dse",
        "-early-cse",
        "-functionattrs",
        "-functionattrs",
        "-globaldce",
        "-globalopt",
        "-gvn",
        "-indvars",
        "-inline",
        "-instcombine",
        "-ipsccp",
        "-jump-threading",
        "-lcssa",
        "-licm",
        "-loop-deletion",
        "-loop-idiom",
        "-loop-reduce",
        "-loop-rotate",
        "-loop-simplify",
        "-loop-unroll",
        "-loop-unswitch",
        "-lower-expect",
        "-loweratomic",
        "-lowerinvoke",
        "-lowerswitch",
        "-mem2reg",
        "-memcpyopt",
        "-partial-inliner",
        "-prune-eh",
        "-reassociate",
        "-sccp",
        "-simplifycfg",
        "-sink",
        "-sroa",
        "-strip",
        "-strip-nondebug",
        "-tailcallelim",
    ]

    def __init__(self, env, flags=None):
        super().__init__(env=env)
        # Array for translating from this tiny action space to the action space of
        # the wrapped environment.
        self.true_action_indices = [self.action_space[f] for f in self.action_space_subset]

    def action(self, action: tp.Union[int, tp.List[int]]):
        if isinstance(action, int):
            return self.true_action_indices[action]
        else:
            return [self.true_action_indices[a] for a in action]


class CompilerGym(ExperimentFunction):
    def __init__(self, pb_index: int):
        action_space_size = len(SmallActionSpaceLlvmEnv.action_space_subset)
        self.num_episode_steps = 45
        parametrization = (
            ng.p.Array(shape=(self.num_episode_steps,)).set_bounds(0, action_space_size - 1).set_integer_casting()
        ).set_name("direct")
        env = gym.make("llvm-ic-v0", observation_space="Autophase", reward_space="IrInstructionCountOz")
        self.uris = list(env.datasets["benchmark://cbench-v1"].benchmark_uris())
        self.compilergym_index = pb_index
        env.reset(benchmark=self.compilergym_index)
        super().__init__(self.eval_actions_as_list, parametrization=parametrization)

    def make_env(self) -> gym.Env:
        """Convenience function to create the environment that we'll use."""
        # User the time-limited wrapper to fix the length of episodes.
        env = gym.wrappers.TimeLimit(
            env=SmallActionSpaceLlvmEnv(env=gym.make("llvm-v0", reward_space="IrInstructionCountOz")),
            max_episode_steps=self.num_episode_steps,
        )
        env.require_dataset("cBench-v1")
        env.unwrapped.benchmark = "cBench-v1/qsort"
        return env

    # @lru_cache(maxsize=1024)  # function is deterministic so we can cache results
    def eval_actions(self, actions: tp.Tuple[int]) -> float:
        """Create an environment, run the sequence of actions in order, and return the
        negative cumulative reward. Intermediate observations/rewards are discarded.

        This is the function that we want to minimize.
        """
        with self.make_env() as env:
            env.reset(benchmark=self.compilergym_index)
            _, reward, _, _ = env.step(actions)
        return -env.episode_reward

    def eval_actions_as_list(self, actions: tp.List[int]):
        """Wrapper around eval_actions() that records the return value for later analysis."""
        action_space_size = len(SmallActionSpaceLlvmEnv.action_space_subset)
        reward = self.eval_actions(tuple([actions[i] for i in range(len(actions))]))
        # action_names = [SmallActionSpaceLlvmEnv.action_space_subset[a] for a in actions]
        # print(len(rewards_list), f"{-reward:.6f}", " ".join(action_names), sep='\t')
        return reward


class GymMulti(ExperimentFunction):

    env_names = GYM_ENV_NAMES

    controllers = CONTROLLERS

    ng_gym = [
        "Copy-v0",
        "RepeatCopy-v0",
        "Reverse-v0",
        "CartPole-v0",
        "CartPole-v1",
        "Acrobot-v1",
        "FrozenLake-v0",
        "FrozenLake8x8-v0",
        "NChain-v0",
        "Roulette-v0",
    ]

    def __init__(
        self,
        name: str = "gym_anm:ANM6Easy-v0",
        control: str = "conformant",
        neural_factor: int = 2,
        randomized: bool = True,
    ) -> None:
        if os.name == "nt":
            raise ng.errors.UnsupportedExperiment("Windows is not supported")
        if "compilergym" in name:
            env = gym.make("llvm-ic-v0", observation_space="Autophase", reward_space="IrInstructionCountOz")
            self.uris = list(env.datasets["benchmark://cbench-v1"].benchmark_uris())
            self.csmith = list(env.datasets["generator://csmith-v0"].benchmark_uris())[:100]

            if "stoc" in name:
                self.compilergym_index = None
                o = env.reset(benchmark=np.random.choice(self.csmith))
            else:
                self.compilergym_index = np.random.choice(self.uris)
                o = env.reset(benchmark=self.compilergym_index)
            # env.require_dataset("cBench-v1")
            # env.unwrapped.benchmark = "benchmark://cBench-v1/qsort"
        else:
            env = gym.make(name if "LANM" not in name else "gym_anm:ANM6Easy-v0")
            o = env.reset()
        self.env = env

        # Build various attributes.
        self.name = (
            (name if not "compiler" in name else name + str(env)) + "__" + control + "__" + str(neural_factor)
        )
        if randomized:
            self.name += "_unseeded"
        self.randomized = randomized
        try:
            self.num_time_steps = env._max_episode_steps  # I know! This is a private variable.
        except AttributeError:  # Not all environements have a max number of episodes!
            assert any(x in name for x in NO_LENGTH), name
            self.num_time_steps = 45 if "ompiler" in name else (100 if "LANM" not in name else 3000)
        self.gamma = 0.995 if "LANM" in name else 1.0
        self.neural_factor = neural_factor

        # Infer the action space.
        if isinstance(env.action_space, gym.spaces.Discrete):
            output_dim = env.action_space.n
            output_shape = (output_dim,)
            discrete = True
            assert output_dim is not None, env.action_space.n
        else:  # Continuous action space
            output_shape = env.action_space.shape
            if output_shape is None:
                output_shape = tuple(np.asarray(env.action_space.sample()).shape)  # type: ignore
            # When the shape is not available we might do:
            # output_shape = tuple(np.asarray(env.action_space.sample()).shape)  # type: ignore
            discrete = False
            output_dim = np.prod(output_shape)
        self.discrete = discrete

        # Infer the observation space.
        assert (
            env.observation_space is not None or "ompiler" in name
        ), "An observation space should be defined."
        if "ompiler" in self.name:
            input_dim = 56
            self.discrete_input = False
        elif env.observation_space is not None and env.observation_space.dtype == int:
            # Direct inference for corner cases:
            # if "int" in str(type(o)):
            input_dim = env.observation_space.n
            assert input_dim is not None, env.observation_space.n
            self.discrete_input = True
        else:
            input_dim = np.prod(env.observation_space.shape) if env.observation_space is not None else 0
            if input_dim is None:
                input_dim = np.prod(np.asarray(o).shape)
            self.discrete_input = False

        # Infer the action type.
        a = env.action_space.sample()
        self.action_type = type(a)
        self.subaction_type = None
        if hasattr(a, "__iter__"):
            self.subaction_type = type(a[0])

        # Prepare the policy shape.
        self.output_shape = output_shape
        self.num_stacking = 1
        self.memory_len = neural_factor * input_dim if "memory" in control else 0
        self.extended_input_len = (input_dim + output_dim) * self.num_stacking if "stacking" in control else 0
        input_dim = input_dim + self.memory_len + self.extended_input_len
        self.extended_input = np.zeros(self.extended_input_len)
        output_dim = output_dim + self.memory_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_neurons = neural_factor * (input_dim - self.extended_input_len)
        self.num_internal_layers = 1 if "semi" in control else 3
        internal = self.num_internal_layers * (self.num_neurons ** 2) if "deep" in control else 0
        unstructured_neural_size = (
            output_dim * self.num_neurons + self.num_neurons * (input_dim + 1) + internal,
        )
        neural_size = unstructured_neural_size
        assert control in CONTROLLERS or control == "conformant", f"{control} not known as a form of control"
        self.control = control
        if "neural" in control:
            self.first_size = self.num_neurons * (self.input_dim + 1)
            self.second_size = self.num_neurons * self.output_dim
            self.first_layer_shape = (self.input_dim + 1, self.num_neurons)
            self.second_layer_shape = (self.num_neurons, self.output_dim)
        shape_dict = {
            "conformant": (self.num_time_steps,) + output_shape,
            "stochastic_conformant": (self.num_time_steps,) + output_shape,
            "linear": (input_dim + 1, output_dim),
            "memory_neural": neural_size,
            "neural": neural_size,
            "deep_neural": neural_size,
            "semideep_neural": neural_size,
            "deep_memory_neural": neural_size,
            "semideep_memory_neural": neural_size,
            "deep_stackingmemory_neural": neural_size,
            "stackingmemory_neural": neural_size,
            "semideep_stackingmemory_neural": neural_size,
            "deep_extrapolatestackingmemory_neural": neural_size,
            "extrapolatestackingmemory_neural": neural_size,
            "semideep_extrapolatestackingmemory_neural": neural_size,
            "structured_neural": neural_size,
            "multi_neural": (min(self.num_time_steps, 50),) + unstructured_neural_size,
            "noisy_neural": neural_size,
            "noisy_scrambled_neural": neural_size,
            "scrambled_neural": neural_size,
        }
        shape = shape_dict[control]
        assert all(
            c in shape_dict for c in self.controllers
        ), f"{self.controllers} subset of {shape_dict.keys()}"
        shape = tuple(map(int, shape))
        self.policy_shape = shape if "structured" not in control else None

        # Create the parametrization.
        parametrization = parameter.Array(shape=shape).set_name("ng_default")
        if "structured" in control and "neural" in control and "multi" not in control:
            parametrization = parameter.Instrumentation(  # type: ignore
                parameter.Array(shape=tuple(map(int, self.first_layer_shape))),
                parameter.Array(shape=tuple(map(int, self.second_layer_shape))),
            ).set_name("ng_struct")
        if "conformant" in control:
            try:
                if env.action_space.low is not None and env.action_space.high is not None:
                    low = np.repeat(np.expand_dims(env.action_space.low, 0), self.num_time_steps, axis=0)
                    high = np.repeat(np.expand_dims(env.action_space.high, 0), self.num_time_steps, axis=0)
                    init = 0.5 * (low + high)
                    parametrization = parameter.Array(init=init)
                    parametrization.set_bounds(low, high)
            except AttributeError:  # Not all env.action_space have a low and a high.
                pass
            if self.subaction_type == int:
                parametrization.set_integer_casting()
            parametrization.set_name("conformant")

        # Now initializing.
        super().__init__(self.gym_multi_function, parametrization=parametrization)
        self.parametrization.function.deterministic = False
        self.archive: tp.List[tp.Any] = []
        self.mean_loss = 0.0
        self.num_losses = 0

    def evaluation_function(self, *recommendations) -> float:
        """Averages multiple evaluatioons if necessary."""
        x = recommendations[0].value
        if not self.randomized:
            return self.gym_multi_function(x, limited_fidelity=False)
        losses = [
            self.gym_multi_function(x, limited_fidelity=False, pb_index=pb_index) for pb_index in range(23)
        ]
        return sum(losses) / len(losses)

    def discretize(self, a):
        """Transforms a logit into an int obtained through softmax."""
        probabilities = np.exp(a - max(a))
        probabilities = probabilities / sum(probabilities)
        return int(list(np.random.multinomial(1, probabilities)).index(1))

    def neural(self, x: np.ndarray, o: np.ndarray):
        """Applies a neural net parametrized by x to an observation o."""
        o = o.ravel()
        if self.control == "linear":
            # The linear case is simplle.
            output = np.matmul(o, x[1:, :])
            output += x[0]
            return output.reshape(self.output_shape), np.zeros(0)
        if "structured" not in self.control:
            # If not structured then we split into two matrices.
            first_matrix = x[: self.first_size].reshape(self.first_layer_shape) / np.sqrt(len(o))
            second_matrix = x[self.first_size : (self.first_size + self.second_size)].reshape(
                self.second_layer_shape
            ) / np.sqrt(self.num_neurons)
        else:
            # In the structured case we should have two entries with the right shapes.
            assert len(x) == 2
            first_matrix = np.asarray(x[0][0])
            second_matrix = np.asarray(x[0][1])
            assert (
                first_matrix.shape == self.first_layer_shape
            ), f"{first_matrix} does not match {self.first_layer_shape}"
            assert (
                second_matrix.shape == self.second_layer_shape
            ), f"{second_matrix} does not match {self.second_layer_shape}"
        assert len(o) == len(first_matrix[1:]), f"{o.shape} coming in matrix of shape {first_matrix.shape}"
        output = np.matmul(o, first_matrix[1:])
        if "deep" in self.control:
            # The deep case must be split into several layers.
            current_index = self.first_size + self.second_size
            internal_layer_size = self.num_neurons ** 2
            s = (self.num_neurons, self.num_neurons)
            for _ in range(self.num_internal_layers):
                output = np.tanh(output)
                output = np.matmul(
                    output, x[current_index : current_index + internal_layer_size].reshape(s)
                ) / np.sqrt(self.num_neurons)
                current_index += internal_layer_size
            assert current_index == len(x)
        output = np.matmul(np.tanh(output + first_matrix[0]), second_matrix)
        return output[self.memory_len :].reshape(self.output_shape), output[: self.memory_len]

    def gym_multi_function(self, x: np.ndarray, limited_fidelity: bool = False, pb_index: int = -1):
        """Do a simulation with parametrization x and return the result."""
        # Deterministic conformant: do  the average of 7 simullations always with the same seed.
        # Otherwise: apply a random seed and do a single simulation.
        num_simulations = 7 if self.control != "conformant" and not self.randomized else 1
        loss = 0
        for seed in range(num_simulations):
            loss += self.gym_simulate(
                x,
                seed=seed if not self.randomized else self.parametrization.random_state.randint(500000),
                limited_fidelity=limited_fidelity,
                pb_index=pb_index,
            )
        return loss / num_simulations

    def action_cast(self, a):
        """Transforms an action into an active as expected by the gym step function."""
        env = self.env
        if type(a) == np.float64:
            a = np.asarray((a,))
        if self.discrete:
            a = self.discretize(a)
        else:
            if type(a) != self.action_type:  # , f"{a} does not have type {self.action_type}"
                a = self.action_type(a)
            try:
                if env.action_space.low is not None and env.action_space.high is not None:
                    # Projection to [0, 1]
                    a = 0.5 * (1.0 + np.tanh(a))
                    # Projection to the right space.
                    a = env.action_space.low + (env.action_space.high - env.action_space.low) * a
            except AttributeError:
                pass  # Sometimes an action space has no low and no high.
            if self.subaction_type is not None:
                if type(a) == tuple:
                    a = tuple(int(_a + 0.5) for _a in a)
                else:
                    for i in range(len(a)):
                        a[i] = self.subaction_type(a[i])
        assert type(a) == self.action_type, f"{a} should have type {self.action_type} "
        try:
            assert env.action_space.contains(
                a
            ), f"In {self.name}, high={env.action_space.high} low={env.action_space.low} {a} is not sufficiently close to {[env.action_space.sample() for _ in range(10)]}"
        except AttributeError:
            pass  # Not all env can do "contains".
        return a

    def step(self, a):
        """Apply an action.

        We have a step on top of Gym's step for storing some statistics."""
        o, r, done, info = self.env.step(a)
        return o, r, done, info

    def heuristic(self, o, current_observations):
        current_observations = np.asarray(current_observations + [o], dtype=np.float32)
        self.archive = [
            self.archive[i] for i in range(len(self.archive)) if self.archive[i][2] <= self.mean_loss
        ]
        self.archive = sorted(self.archive, key=lambda trace: -len(trace[0]))
        for trace in self.archive:
            to, ta, _ = trace
            assert len(to) == len(ta)
            if len(current_observations) > len(to) and "extrapolate" not in self.control:
                continue
            to = np.asarray(to[(-len(current_observations)) :], dtype=np.float32)
            # if all((_to - _o) for _to, _o in zip(to, current_observations)) <= 1e-7:
            if np.array_equal(to, current_observations):
                return np.asarray(ta[len(current_observations) - 1], dtype=np.float32)
        return None

    def gym_simulate(self, x: np.ndarray, seed: int, pb_index: int, limited_fidelity: bool = True):
        """Single simulation with parametrization x."""
        current_time_index = 0
        current_reward = 0.
        current_observations: tp.List[tp.Any] = []
        current_actions: tp.List[tp.Any] = []
        try:
            if self.policy_shape is not None:
                x = x.reshape(self.policy_shape)
        except:
            assert False, f"x has shape {x.shape} and needs {self.policy_shape} for control {self.control}"
        assert seed == 0 or self.control != "conformant" or self.randomized
        env = self.env
        env.seed(seed=seed)
        if "compilergym" in self.name:
            if "stoc" in self.name:
                o = env.reset(
                    benchmark=np.random.choice(self.csmith) if pb_index < 0 else self.uris[pb_index]
                )
            else:
                o = env.reset(benchmark=self.compilergym_index)
        else:
            o = env.reset()
        control = self.control
        if (
            "conformant" in control
        ):  # Conformant planning: we just optimize a sequence of actions. No reactivity.
            return self.gym_conformant(x)
        if "scrambled" in control:  # We shuffle the variables, typically so that progressive methods optimize
            # everywhere in parallel instead of focusing on one single layer for years.
            np.random.RandomState(1234).shuffle(x)
        if "noisy" in control:  # We add a randomly chosen but fixed perturbation of the x, i.e. we do not
            # start at 0.
            x = x + 0.01 * np.random.RandomState(1234).normal(size=x.shape)
        reward = 0.0
        memory = np.zeros(self.memory_len)
        for i in range(self.num_time_steps):
            # Actual loop over time steps!
            if self.discrete_input:
                obs = np.zeros(shape=self.input_dim - self.extended_input_len - len(memory))
                obs[o] = 1
                o = obs
            previous_o = np.asarray(o)
            o = np.concatenate([previous_o.ravel(), memory.ravel(), self.extended_input])
            assert (
                len(o) == self.input_dim
            ), f"o has shape {o.shape} whereas input_dim={self.input_dim} ({control} / {env})"
            a, memory = self.neural(x[i % len(x)] if "multi" in control else x, o)
            a = self.action_cast(a)
            try:
                o, r, done, _ = self.step(a)  # Outputs = observation, reward, done, info.
                current_time_index += 1
                if "multifidLANM" in self.name and current_time_index > 500 and limited_fidelity:
                    done = True
                current_reward *= self.gamma
                current_reward += r
                current_observations += [np.asarray(o).copy()]
                current_actions += [np.asarray(a).copy()]
                if (
                    done and "stacking" in self.control
                ):  # Only the method which do a stacking of heuristic + memory into the
                    # observation need archiving.
                    self.num_losses += 1
                    tau = 1.0 / self.num_losses
                    self.mean_loss = (
                        ((1.0 - tau) * self.mean_loss + tau * current_reward)
                        if self.mean_loss is not None
                        else current_reward
                    )
                    found = False
                    for trace in self.archive:
                        to, _, _ = trace
                        if np.array_equal(
                            np.asarray(current_observations, dtype=np.float32),
                            np.asarray(to, dtype=np.float32),
                        ):
                            found = True
                            break
                    if not found:
                        # Risky: this code assumes that the object is used only in a single run.
                        self.archive += [(current_observations, current_actions, current_reward)]
            except AssertionError:  # Illegal action.
                return 1e20 / (1.0 + i)  # We encourage late failures rather than early failures.
            if "stacking" in control:
                attention_a = self.heuristic(
                    o, current_observations
                )  # Best so far, or something like that heuristically derived.
                a = attention_a if attention_a is not None else 0.0 * np.asarray(a)
                previous_o = previous_o.ravel()
                additional_input = np.concatenate([np.asarray(a).ravel(), previous_o])
                shift = len(additional_input)
                self.extended_input[: (len(self.extended_input) - shift)] = self.extended_input[shift:]
                self.extended_input[(len(self.extended_input) - shift) :] = additional_input
            reward += r
            if done:
                break
        return -reward

    def gym_conformant(self, x: np.ndarray):
        """Conformant: we directly optimize inputs, not parameters of a policy."""
        reward = 0.0
        for i, a in enumerate(10.0 * x):
            a = self.action_cast(a)
            try:
                _, r, done, _ = self.step(a)  # Outputs = observation, reward, done, info.
            except AssertionError:  # Illegal action.
                return 1e20 / (1.0 + i)  # We encourage late failures rather than early failures.
            reward *= self.gamma
            reward += r
            if done:
                break
        # env.render()  if you want to display.
        return -reward
