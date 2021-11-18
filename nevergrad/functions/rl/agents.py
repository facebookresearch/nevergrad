# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import warnings
import operator
import copy as _copy
import typing as tp
import gym
import numpy as np
from nevergrad.common.tools import pytorch_import_fix
from nevergrad.parametrization import parameter as p
from ..base import ExperimentFunction
from . import base
from . import envs

pytorch_import_fix()

# pylint: disable=wrong-import-position,wrong-import-order
import torch as torch  # noqa
import torch.nn.functional as F  # noqa
from torch import nn  # noqa
from torch.utils.data import WeightedRandomSampler  # noqa


class RandomAgent(base.Agent):
    """Agent that plays randomly."""

    def __init__(self, env: gym.Env) -> None:
        self.env = env
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self.num_outputs = env.action_space.n

    def act(
        self,
        observation: tp.Any,
        reward: tp.Any,
        done: bool,
        info: tp.Optional[tp.Dict[tp.Any, tp.Any]] = None,
    ) -> tp.Any:
        return np.random.randint(self.num_outputs)

    def copy(self) -> "RandomAgent":
        return self.__class__(self.env)


class Agent007(base.Agent):
    """Agents that plays slighlty better than random on the 007 game."""

    def __init__(self, env: gym.Env) -> None:
        self.env = env
        assert isinstance(env, envs.DoubleOSeven) or (
            isinstance(env, base.SingleAgentEnv) and isinstance(env.env, envs.DoubleOSeven)
        )

    def act(
        self,
        observation: tp.Any,
        reward: tp.Any,
        done: bool,
        info: tp.Optional[tp.Dict[tp.Any, tp.Any]] = None,
    ) -> tp.Any:
        my_amm, my_prot, their_amm, their_prot = observation  # pylint: disable=unused-variable
        if their_prot == 4 and my_amm:
            action = "fire"
        elif their_amm == 0:
            action = np.random.choice(["fire", "reload"])
        else:
            action = np.random.choice(["fire", "protect", "reload"])
        return envs.JamesBond.actions.index(action)

    def copy(self) -> "Agent007":
        return self.__class__(self.env)


class TorchAgent(base.Agent):
    """Agents than plays through a torch neural network"""

    def __init__(
        self, module: nn.Module, deterministic: bool = True, instrumentation_std: float = 0.1
    ) -> None:
        super().__init__()
        self.deterministic = deterministic
        self.module = module
        kwargs = {
            name: p.Array(shape=value.shape)
            .set_mutation(sigma=instrumentation_std)
            .set_bounds(-10, 10, method="arctan")
            for name, value in module.state_dict().items()  # type: ignore
        }  # bounded to avoid overflows
        self.instrumentation = p.Instrumentation(**kwargs)

    @classmethod
    def from_module_maker(
        cls,
        env: gym.Env,
        module_maker: tp.Callable[[tp.Tuple[int, ...], int], nn.Module],
        deterministic: bool = True,
    ) -> "TorchAgent":
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert isinstance(env.observation_space, gym.spaces.Box)
        module = module_maker(env.observation_space.shape, env.action_space.n)
        return cls(module, deterministic=deterministic)

    def act(
        self,
        observation: tp.Any,
        reward: tp.Any,
        done: bool,
        info: tp.Optional[tp.Dict[tp.Any, tp.Any]] = None,
    ) -> tp.Any:
        obs = torch.from_numpy(observation.astype(np.float32))
        forward = self.module.forward(obs)  # type: ignore
        probas = F.softmax(forward, dim=0)
        if self.deterministic:
            return probas.max(0)[1].view(1, 1).item()
        else:
            return next(iter(WeightedRandomSampler(probas, 1)))

    def copy(self) -> "TorchAgent":
        return TorchAgent(_copy.deepcopy(self.module), self.deterministic)

    def load_state_dict(self, state_dict: tp.Dict[str, np.ndarray]) -> None:
        # pylint: disable=not-callable
        self.module.load_state_dict({x: torch.tensor(y.astype(np.float32)) for x, y in state_dict.items()})


class TorchAgentFunction(ExperimentFunction):
    """Instrumented function which plays the agent using an environment runner"""

    _num_test_evaluations = 1000

    def __init__(
        self,
        agent: TorchAgent,
        env_runner: base.EnvironmentRunner,
        reward_postprocessing: tp.Callable[[float], float] = operator.neg,
    ) -> None:
        assert isinstance(env_runner.env, gym.Env)
        self.agent = agent.copy()
        self.runner = env_runner.copy()
        self.reward_postprocessing = reward_postprocessing
        super().__init__(self.compute, self.agent.instrumentation.copy().set_name(""))
        self.parametrization.function.deterministic = False
        self.add_descriptors(
            num_repetitions=self.runner.num_repetitions, archi=self.agent.module.__class__.__name__
        )

    def compute(self, **kwargs: np.ndarray) -> float:
        self.agent.load_state_dict(kwargs)
        try:  # safeguard against nans
            with torch.no_grad():
                reward = self.runner.run(self.agent)

        except RuntimeError as e:
            warnings.warn(f"Returning 0 after error: {e}")
            reward = 0.0
        assert isinstance(reward, (int, float))
        return self.reward_postprocessing(reward)

    def evaluation_function(self, *recommendations: p.Parameter) -> float:
        """Implements the call of the function.
        Under the hood, __call__ delegates to oracle_call + add some noise if noise_level > 0.
        """
        assert len(recommendations) == 1, "Should not be a pareto set for a singleobjective function"
        num_tests = max(1, int(self._num_test_evaluations / self.runner.num_repetitions))
        return sum(self.compute(**recommendations[0].kwargs) for _ in range(num_tests)) / num_tests


class Perceptron(nn.Module):
    def __init__(self, input_shape: tp.Tuple[int, ...], output_size: int) -> None:
        super().__init__()  # type: ignore
        assert len(input_shape) == 1
        self.head = nn.Linear(input_shape[0], output_size)  # type: ignore

    def forward(self, *args: tp.Any) -> tp.Any:
        assert len(args) == 1
        return self.head(args[0])


class DenseNet(nn.Module):
    def __init__(self, input_shape: tp.Tuple[int, ...], output_size: int) -> None:
        super().__init__()  # type: ignore
        assert len(input_shape) == 1
        self.lin1 = nn.Linear(input_shape[0], 16)  # type: ignore
        self.lin2 = nn.Linear(16, 16)  # type: ignore
        self.lin3 = nn.Linear(16, 16)  # type: ignore
        self.head = nn.Linear(16, output_size)  # type: ignore

    def forward(self, *args: tp.Any) -> tp.Any:
        assert len(args) == 1
        x = F.relu(self.lin1(args[0]))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return self.head(x)
