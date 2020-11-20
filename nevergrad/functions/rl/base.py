# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
from collections import defaultdict
import gym


StepReturn = tp.Tuple[
    tp.Dict[str, tp.Any],
    tp.Dict[str, int],
    tp.Dict[str, bool],
    tp.Dict[str, tp.Any]
]  # ray-like multi-agent return type


class StepOutcome:
    """Handle for dealing with environment (and especially multi-agent) outputs more easily
    """

    def __init__(
        self,
        observation: tp.Any,
        reward: tp.Any = None,
        done: bool = False,
        info: tp.Optional[tp.Dict[tp.Any, tp.Any]] = None
    ) -> None:
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info: tp.Dict[tp.Any, tp.Any] = {} if info is None else info

    def __iter__(self) -> tp.Iterable[tp.Any]:
        return iter((self.observation, self.reward, self.done, self.info))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(observation={self.observation}, reward={self.reward}, done={self.done}, info={self.info})"

    @classmethod
    def from_multiagent_step(
        cls, obs: tp.Dict[str, tp.Any], reward: tp.Dict[str, tp.Any], done: tp.Dict[str, bool], info: tp.Dict[str, tp.Dict[tp.Any, tp.Any]]
    ) -> tp.Tuple[tp.Dict[str, "StepOutcome"], bool]:
        outcomes = {
            agent: cls(obs[agent], reward.get(agent, None), done.get(agent, done.get("__all__", False)), info.get(agent, {}))
            for agent in obs
        }
        return outcomes, done.get("__all__", False)

    @staticmethod
    def to_multiagent_step(outcomes: tp.Dict[str, "StepOutcome"], done: bool = False) -> StepReturn:
        names = ["observation", "reward", "done", "info"]
        obs, reward, done_dict, info = ({agent: getattr(outcome, name) for agent, outcome in outcomes.items()} for name in names)
        done_dict["__all__"] = done
        return obs, reward, done_dict, info


class Agent:
    """Base class for an Agent operating in an environment.
    """

    def act(self, observation: tp.Any, reward: tp.Any, done: bool, info: tp.Optional[tp.Dict[tp.Any, tp.Any]] = None) -> tp.Any:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def copy(self) -> "Agent":
        return self.__class__()


class MultiAgentEnv:
    """Base class for an multi-agent environment (in a ray-like fashion).
    """

    observation_space: gym.Space
    action_space: gym.Space
    agent_names: tp.List[str]

    def reset(self) -> tp.Dict[str, tp.Any]:
        raise NotImplementedError

    def step(self, action_dict: tp.Dict[str, tp.Any]) -> StepReturn:
        """Returns observations from ready agents.
        The returns are dicts mapping from agent_id strings to StepOutcome. The
        number of agents in the env can vary over time.
        """
        raise NotImplementedError

    def copy(self) -> "MultiAgentEnv":
        """Used to create new instances of Ray MultiAgentEnv
        """
        raise NotImplementedError

    def with_agent(self, **agents: Agent) -> "PartialMultiAgentEnv":
        return PartialMultiAgentEnv(self, **agents)


class PartialMultiAgentEnv(MultiAgentEnv):
    """Multi agent environement for which some of the agents have been fixed
    """

    def __init__(self, env: MultiAgentEnv, **agents: Agent) -> None:
        self.env = env.copy()
        self.agents = {name: agent.copy() for name, agent in agents.items()}
        self.env.reset()
        unknown = set(agents) - set(env.agent_names)
        if unknown:  # this assumes that all agents play from the start
            raise ValueError(f"Unkwnon agents: {unknown}")
        self.agent_names = [an for an in env.agent_names if an not in agents]
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._agents_outcome: tp.Dict[str, StepOutcome] = {}

    def reset(self) -> tp.Dict[str, tp.Any]:
        outcomes = StepOutcome.from_multiagent_step(self.env.reset(), {}, {}, {})[0]
        self._agents_outcome = {name: outcomes[name] for name in self.agents}
        return {name: outcomes[name].observation for name in outcomes if name not in self.agents}

    def step(self, action_dict: tp.Dict[str, tp.Any]) -> StepReturn:
        """Returns observations from ready agents.
        The returns are dicts mapping from agent_id strings to StepOutcome. The
        number of agents in the env can vary over time.
        """
        full_action_dict = {name: self.agents[name].act(*outcome) for name, outcome in self._agents_outcome.items()}  # type: ignore
        full_action_dict.update(action_dict)
        outcomes, done = StepOutcome.from_multiagent_step(*self.env.step(full_action_dict))
        self._agents_outcome = {name: outcomes[name] for name in self.agents}
        return StepOutcome.to_multiagent_step({name: outcomes[name] for name in outcomes if name not in self.agents}, done)

    def copy(self) -> "PartialMultiAgentEnv":
        return self.__class__(self.env, **self.agents)

    def as_single_agent(self) -> "SingleAgentEnv":
        return SingleAgentEnv(self)


# pylint: disable=abstract-method
class SingleAgentEnv(gym.Env):  # type: ignore
    """Single-agent gym-like environment based on a multi-agent environment for which
    all but one agent has been fixed.
    """

    def __init__(self, env: PartialMultiAgentEnv):
        assert len(env.agent_names) == 1, f"Too many remaining agents: {self.agent_names}"
        self.env = env.copy()
        self.env.reset()
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._agent_name = env.agent_names[0]
        self.env = env

    def reset(self) -> tp.Any:
        return self.env.reset()[self._agent_name]

    def step(self, action: tp.Any) -> tp.Tuple[tp.Any, tp.Any, bool, tp.Dict[tp.Any, tp.Any]]:
        an = self._agent_name
        obs, reward, done, info = self.env.step({an: action})
        return obs[an], reward[an], done[an] | done["__all__"], info.get(an, {})

    def copy(self) -> "SingleAgentEnv":
        return self.__class__(self.env.copy())


class EnvironmentRunner:
    """Helper for running environements

    Parameters
    ----------
    env: gym.Env or MultiAgentEnv
        a possibly multi agent environment
    num_repetations: int
        number of repetitions to play the environment (smoothes the output)
    max_step: int
        maximum number of steps to play the environemnet before breaking
    """

    def __init__(
            self,
            env: tp.Union[gym.Env, MultiAgentEnv],
            num_repetitions: int = 1,
            max_step: float = float("inf")
    ) -> None:
        self.env = env
        self.num_repetitions = num_repetitions
        self.max_step = max_step

    def run(self, *agent: Agent, **agents: Agent) -> tp.Union[float, tp.Dict[str, float]]:
        """Run one agent or multiple named agents

        Parameters
        ----------
        *agent: Agent (optional)
            the agent to play a single-agent environment
        **agents: Agent
            the named agents to play a multi-agent environment

        Returns
        -------
        float:
            the mean reward (possibly for each agent)
        """
        san = "single_agent_name"
        sum_rewards: tp.Dict[str, float] = {name: 0.0 for name in agents} if agents else {san: 0.0}
        for _ in range(self.num_repetitions):
            rewards = self._run_once(*agent, **agents)
            for name, value in rewards.items():
                sum_rewards[name] += value
        mean_rewards = {name: float(value) / self.num_repetitions for name, value in sum_rewards.items()}
        if isinstance(self.env, gym.Env):
            return mean_rewards[san]
        return mean_rewards

    def _run_once(self, *single_agent: Agent, **agents: Agent) -> tp.Dict[str, float]:
        san = "single_agent_name"
        if len(single_agent) == 1 and not agents:
            agents = {san: single_agent[0]}
        elif single_agent or not agents:
            raise ValueError("Either provide 1 unnamed agent or several named agents")
        for agent in agents.values():
            agent.reset()
        if isinstance(self.env, gym.Env):
            outcomes, done = {san: StepOutcome(self.env.reset())}, False
        else:
            outcomes, done = StepOutcome.from_multiagent_step(self.env.reset(), {}, {}, {})
        reward_sum: tp.Dict[str, float] = defaultdict(float)
        step = 0
        while step < self.max_step and not done:
            actions: tp.Dict[str, tp.Any] = {}
            for name, outcome in outcomes.items():
                actions[name] = agents[name].act(*outcome)  # type: ignore
            if isinstance(self.env, gym.Env):
                outcomes = {san: StepOutcome(*self.env.step(actions[san]))}
                done = outcomes[san].done
            else:
                outcomes, done = StepOutcome.from_multiagent_step(*self.env.step(actions))
            for name, outcome in outcomes.items():
                assert outcome.reward is not None
                reward_sum[name] += outcome.reward
            step += 1
        for name, outcome in outcomes.items():
            agents[name].act(*outcome)  # type: ignore
        return reward_sum

    def copy(self) -> "EnvironmentRunner":
        return EnvironmentRunner(self.env.copy(), num_repetitions=self.num_repetitions, max_step=self.max_step)
