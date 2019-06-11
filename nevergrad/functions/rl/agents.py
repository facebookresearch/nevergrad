import warnings
import operator
from typing import Dict, Any, Optional, Callable, Tuple, Type
import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import WeightedRandomSampler
from nevergrad import instrumentation as inst
from nevergrad.functions import utils
from nevergrad.optimization.base import Optimizer
from . import base
from . import envs


class RandomAgent(base.Agent):
    """Agent that plays randomly.
    """

    def __init__(self, env: gym.Env) -> None:
        self.env = env
        assert isinstance(env.action_space, gym.spaces.Discrete)
        self.num_outputs = env.action_space.n

    def act(self, observation: Any, reward: Any, done: bool, info: Optional[Dict[Any, Any]] = None) -> Any:
        return np.random.randint(self.num_outputs)

    def duplicate(self) -> "RandomAgent":
        return self.__class__(self.env)


class Agent007(base.Agent):
    """Agents that plays slighlty better than random on the 007 game.
    """

    def __init__(self, env: gym.Env) -> None:
        self.env = env
        assert isinstance(env, envs.DoubleOSeven) or (isinstance(env, base.SingleAgentEnv) and isinstance(env.env, envs.DoubleOSeven))

    def act(self, observation: Any, reward: Any, done: bool, info: Optional[Dict[Any, Any]] = None) -> Any:
        my_amm, my_prot, their_amm, their_prot = observation  # pylint: disable=unused-variable
        if their_prot == 4 and my_amm:
            action = "fire"
        elif their_amm == 0:
            action = np.random.choice(["fire", "reload"])
        else:
            action = np.random.choice(["fire", "protect", "reload"])
        return envs.JamesBond.actions.index(action)

    def duplicate(self) -> "Agent007":
        return self.__class__(self.env)


class TorchAgent(base.Agent):
    """Agents than plays through a torch neural network
    """

    def __init__(self, module: nn.Module, deterministic: bool = True, instrumentation_std: float = 0.1) -> None:
        super().__init__()
        self.deterministic = deterministic
        self.module = module
        kwargs = {
            name: inst.var.Array(*value.shape).affined(a=instrumentation_std).bounded(-10, 10, transform="arctan")
            for name, value in module.state_dict().items()
        }  # bounded to avoid overflows
        self.instrumentation = inst.Instrumentation(**kwargs)

    @classmethod
    def from_module_maker(
        cls, env: gym.Env, module_maker: Callable[[Tuple[int, ...], int], nn.Module], deterministic: bool = True
    ) -> "TorchAgent":
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert isinstance(env.observation_space, gym.spaces.Box)
        module = module_maker(env.observation_space.shape, env.action_space.n)
        return cls(module, deterministic=deterministic)

    def act(self, observation: Any, reward: Any, done: bool, info: Optional[Dict[Any, Any]] = None) -> Any:
        obs = torch.from_numpy(observation.astype(np.float32))
        forward = self.module.forward(obs)
        probas = F.softmax(forward, dim=0)
        if self.deterministic:
            return probas.max(0)[1].view(1, 1).item()
        else:
            return next(iter(WeightedRandomSampler(probas, 1)))

    def duplicate(self) -> "TorchAgent":
        return TorchAgent(self.module, self.deterministic)

    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
        self.module.load_state_dict({x: torch.tensor(y.astype(np.float32)) for x, y in state_dict.items()})


class TrainingTorchAgent(base.Agent):
    """
    Base class for an `Agent` operating in a `gym.Env`
    """

    def __init__(self, agent: TorchAgent, OptimizerClass: Type[Optimizer], num_repetitions: int = 1) -> None:
        super().__init__()
        self.agent = agent
        self.optimizer = OptimizerClass(instrumentation=agent.instrumentation, budget=None, num_workers=1)
        self.num_repetitions = num_repetitions
        self.repetition = 0
        self.cum_reward = 0.0
        self.current_candidate = self.optimizer.ask()
        self.agent.load_state_dict(self.current_candidate.kwargs)

    def act(self, observation: Any, reward: Any, done: bool, info: Optional[Dict[Any, Any]] = None) -> Any:
        if reward is not None:
            self.cum_reward += reward
        return self.agent.act(observation, reward, done, info)

    def reset(self) -> None:
        self.repetition += 1
        if self.repetition >= self.num_repetitions:
            self.optimizer.tell(self.current_candidate, -self.cum_reward / self.repetition)
            self.repetition = 0
            self.current_candidate = self.optimizer.ask()
            self.cum_reward = 0
        self.agent.load_state_dict(self.current_candidate.kwargs)
        self.agent.reset()

    def duplicate(self) -> "TrainingTorchAgent":
        return TrainingTorchAgent(self.agent.duplicate(), self.optimizer.__class__, num_repetitions=self.num_repetitions)


class TorchAgentFunction(inst.InstrumentedFunction, utils.NoisyBenchmarkFunction):
    def __init__(
        self, agent: TorchAgent, env_runner: base.EnvironmentRunner, reward_postprocessing: Callable[[float], float] = operator.neg
    ) -> None:
        assert isinstance(env_runner.env, gym.Env)
        self.agent = agent.duplicate()
        self.runner = env_runner
        self.reward_postprocessing = reward_postprocessing
        super().__init__(self.compute, **self.agent.instrumentation.kwargs)
        self._descriptors.update(num_repetitions=self.runner.num_repetitions, instrumentation="")

    def compute(self, **kwargs: np.ndarray) -> float:
        self.agent.load_state_dict(kwargs)
        try:  # safeguard against nans
            reward = self.runner.run(self.agent)
        except RuntimeError as e:
            warnings.warn(f"Returning 0 after error: {e}")
            reward = 0.0
        assert isinstance(reward, (int, float))
        return self.reward_postprocessing(reward)

    def noisefree_function(self, *args: Any, **kwargs: Any) -> float:
        """Implements the call of the function.
        Under the hood, __call__ delegates to oracle_call + add some noise if noise_level > 0.
        """
        num_tests = int(1000 / self.runner.num_repetitions)  # hardcoded
        return sum(self.compute(**kwargs) for _ in range(num_tests)) / num_tests


class Perceptron(nn.Module):  # type: ignore
    def __init__(self, input_shape: Tuple[int, ...], output_size: int) -> None:
        super().__init__()
        assert len(input_shape) == 1
        self.head = nn.Linear(input_shape[0], output_size)

    def forward(self, *args: Any) -> Any:
        assert len(args) == 1
        return self.head(args[0])


class DenseNet(nn.Module):  # type: ignore
    def __init__(self, input_shape: Tuple[int, ...], output_size: int) -> None:
        super().__init__()
        assert len(input_shape) == 1
        self.lin1 = nn.Linear(input_shape[0], 16)
        self.lin2 = nn.Linear(16, 16)
        self.lin3 = nn.Linear(16, 16)
        self.head = nn.Linear(16, output_size)

    def forward(self, *args: Any) -> Any:
        assert len(args) == 1
        x = F.relu(self.lin1(args[0]))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        return self.head(x)
