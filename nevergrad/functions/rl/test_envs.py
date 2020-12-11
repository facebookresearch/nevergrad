# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp
import pytest
import numpy as np
from nevergrad.common import testing
from . import envs


def test_player() -> None:
    player = envs.JamesBond()
    assert player.get_state() == (0, 0)
    action_states = [("protect", (0, 1)), ("protect", (0, 2)), ("reload", (1, 0)), ("fire", (0, 0))]
    for k, (action, state) in enumerate(action_states):
        player.update_with_action(action)
        assert player.get_state() == state, f"Unexpected state at action #{k}"
    with pytest.raises(ValueError):
        player.update_with_action("fire")
    with pytest.raises(ValueError):
        player.update_with_action("blublu")


def test_doubleoseven_observations() -> None:
    game = envs.DoubleOSeven()
    game.reset()
    obs = game.step({"player_0": 0, "player_1": 1})[
        0
    ]  # p0 fires (reloads since no ammunition) and p1 protects
    expected = {
        "player_0": np.array([1, 0, 0, 1]),  # p0 1 ammu 0 protect, p1 0 ammu 1 protect
        "player_1": np.array([0, 1, 1, 0]),
    }  # p1 then p0
    assert len(obs) == 2
    for k in range(2):
        np.testing.assert_array_equal(obs[f"player_{k}"], expected[f"player_{k}"])


@testing.parametrized(
    first_fire_0=([("reload", "reload"), ("fire", "reload")], {"player_0": 1, "player_1": 0}),
    # first fire is converted to reload for lack of ammunitions:
    first_fire_1=([("fire", "reload"), ("reload", "fire")], {"player_0": 0, "player_1": 1}),
    protections=([("protect", "protect")] * 10 + [("reload", "protect")], {"player_0": 1, "player_1": 0}),
    protections_restart=(
        [("protect", "protect")] * 10 + [("reload", "reload"), ("protect", "fire"), ("fire", "reload")],
        {"player_0": 1, "player_1": 0},
    ),
    max_protections=([("protect", "protect")] * 100, {"player_0": 0, "player_1": 0}),
    double_fire=(
        [("reload", "reload"), ("reload", "reload"), ("fire", "fire"), ("fire", "reload")],
        {"player_0": 1, "player_1": 0},
    ),
)
def test_doubleoseven(base_sequence: tp.List[tp.Tuple[str, str]], base_expected: tp.Dict[str, int]) -> None:
    game = envs.DoubleOSeven()
    for case in ["standard", "symmetric"]:
        # prepare sequence for either standard case or symmetric case (player1 and player2 actions are inverted)
        # this makes sure everything is symmetrical
        if case == "standard":
            sequence = base_sequence
            expected = base_expected
        else:
            sequence = [(a[1], a[0]) for a in base_sequence]
            expected = {"player_0": base_expected["player_1"], "player_1": base_expected["player_0"]}
        # play
        game.reset()
        for k, actions in enumerate(sequence):
            actions_dict = {f"player_{k}": envs.JamesBond.actions.index(a) for k, a in enumerate(actions)}
            _, rew, done, _ = game.step(actions_dict)
            if k != len(sequence) - 1 and done["__all__"]:
                raise AssertionError(
                    f"The game should not have finished at step {k} with actions {actions} ({case})"
                )
        if not done["__all__"]:
            # pylint: disable=undefined-loop-variable
            raise AssertionError(
                f"The game should have finished at last step with actions {actions} ({case})"
            )
        assert rew == expected, f"Wrong output for case: {case}"
