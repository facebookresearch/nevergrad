# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import numpy as np
import gym
from . import base


class JamesBond:
    """Holds the state of a player of DoubleOSeven
    """

    actions = ["fire", "protect", "reload"]
    max_consecutive_protect = 4

    def __init__(self) -> None:
        self.ammunitions = 0
        self.consecutive_protect = 0

    def update_with_action(self, action: str) -> None:
        """Update the current state with an authorized action.
        """
        if action not in self.actions:
            raise ValueError(f"Unauthorized action {action}, authorized are {self.actions}")
        if action == "fire":
            if not self.ammunitions:
                raise ValueError("Cannot fire without amunitions")
            self.ammunitions -= 1
        elif action == "reload":
            self.ammunitions += 1
        if action == "protect":
            self.consecutive_protect += 1
        else:
            self.consecutive_protect = 0

    def get_state(self) -> tp.Tuple[int, int]:
        return (self.ammunitions, self.consecutive_protect)


class DoubleOSeven(base.MultiAgentEnv):
    """2-player environment with "player_0" and "player_1", playing the 007 game.
    Each player state has:
    - a number of ammunition (init: 0)
    - a number of consecutive protection (init: 0)
    Each player can:
    - reload: to get one more ammunition
    - fire: to try to kill the other player (reloads instead if no ammunition)
    - protect: to avoid the other player's fire

    The winner is either:
    - the first to fire while the other people is reloading (or firing without ammunition)
    - the one that did not protect more that 4 times consecutively if the other one did.
    The game stops if nobody won after 100 iterations.
    """

    observation_space = gym.spaces.Box(low=0, high=float("inf"), shape=(4,), dtype=int)
    action_space = gym.spaces.Discrete(3)
    agent_names = ["player_0", "player_1"]

    def __init__(self, verbose: bool = False) -> None:
        super().__init__()
        self.players = [JamesBond(), JamesBond()]
        self.verbose = verbose
        self._step = 0

    def reset(self) -> tp.Dict[str, np.ndarray]:
        """Resets the env and returns observations from ready agents.
        Returns:
            obs (dict): New observations for each ready agent.
        """
        self._step = 0
        self.players = [JamesBond(), JamesBond()]
        return self._make_observations()

    def _make_observations(self) -> tp.Dict[str, np.ndarray]:
        states = [p.get_state() for p in self.players]
        return {"player_0": np.array(states[0] + states[1]), "player_1": np.array(states[1] + states[0])}

    def copy(self) -> "DoubleOSeven":
        return self.__class__(verbose=self.verbose)

    def step(self, action_dict: tp.Dict[str, int]) -> base.StepReturn:
        """Returns observations from ready agents.
        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.
        Returns
        -------
            obs (dict): New observations for each ready agent.
            rewards (dict): Reward values for each ready agent. If the
                episode is just started, the value will be None.
            dones (dict): Done values for each ready agent. The special key
                "__all__" (required) is used to indicate env termination.
            infos (dict): Optional info values for each agent id.
        """
        if self.verbose:
            strings: tp.List[str] = []
            for k in range(2):
                action = JamesBond.actions[action_dict[f"player_{k}"]]
                strings.append(f"Player {k} {self.players[k].get_state()}: {action}")
            print(" - ".join(strings))
        actions = [JamesBond.actions[action_dict[f"player_{k}"]] for k in range(2)]
        self._step += 1
        info: tp.Dict[tp.Any, tp.Any] = {}
        rew = {"player_0": 0, "player_1": 0}
        # change impossible actions
        actions = ["reload" if a == "fire" and not p.ammunitions else a for p, a in zip(self.players, actions)]
        # update players
        for player, action in zip(self.players, actions):
            player.update_with_action(action)
        # main way to win
        if actions[0] == "fire" and actions[1] == "reload":
            rew = {"player_0": 1, "player_1": 0}
        elif actions[0] == "reload" and actions[1] == "fire":
            rew = {"player_0": 0, "player_1": 1}
        # lose if you keep protecting
        if any(p.consecutive_protect > JamesBond.max_consecutive_protect for p in self.players):
            if self.players[0].consecutive_protect > self.players[1].consecutive_protect:
                rew = {"player_0": 0, "player_1": 1}
            elif self.players[1].consecutive_protect > self.players[0].consecutive_protect:
                rew = {"player_0": 1, "player_1": 0}
            # if both keep protecting... well, it goes on...
        obs = self._make_observations()
        done = {"__all__": self._step == 100 or sum(abs(x) for x in rew.values()) > 0}
        return obs, rew, done, info
