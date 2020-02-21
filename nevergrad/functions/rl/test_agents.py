# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
from nevergrad.optimization import optimizerlib
from nevergrad.optimization import helpers
from . import agents
from . import envs
from . import base


def test_play_environment() -> None:
    mgame = envs.DoubleOSeven(verbose=True)
    player_0 = agents.RandomAgent(mgame)
    player_1 = agents.RandomAgent(mgame)
    runner = base.EnvironmentRunner(mgame)
    rewards = runner.run(player_1=player_1, player_0=player_0)
    assert isinstance(rewards, dict)
    assert sum(rewards.values()) in [0, 1]


def test_play_single_agent_environment() -> None:
    mgame = envs.DoubleOSeven()
    game = mgame.with_agent(player_1=agents.RandomAgent(mgame)).as_single_agent()
    rand = agents.RandomAgent(game)
    runner = base.EnvironmentRunner(game)
    reward = runner.run(rand)
    assert isinstance(reward, float)
    assert reward in [0, 1]


def test_torch_agent() -> None:
    mgame = envs.DoubleOSeven()
    game = mgame.with_agent(player_1=agents.RandomAgent(mgame)).as_single_agent()
    obs = game.reset()
    agent = agents.TorchAgent.from_module_maker(game, agents.DenseNet)
    output = agent.act(obs, 0.0, False, None)
    assert output in game.action_space


def test_torch_agent_function() -> None:
    mgame = envs.DoubleOSeven()
    game = mgame.with_agent(player_1=agents.RandomAgent(mgame)).as_single_agent()
    agent = agents.TorchAgent.from_module_maker(game, agents.DenseNet)
    runner = base.EnvironmentRunner(game)
    agentfunction = agents.TorchAgentFunction(agent, runner)
    instru = agentfunction.parametrization
    args, kwargs = instru.spawn_child().set_standardized_data([0] * instru.dimension).value
    assert not args
    value = agentfunction.compute(**kwargs)
    assert value in [0, 1]
    # optimization
    opt = optimizerlib.OnePlusOne(instru, budget=10)
    opt.minimize(agentfunction.compute)


def test_partial_double_seven() -> None:
    mgame = envs.DoubleOSeven()
    mgame.verbose = True
    game = mgame.with_agent(player_1=agents.RandomAgent(mgame)).as_single_agent()
    done = False
    game.reset()
    while not done:
        _, rew, done, _ = game.step(2)
    assert rew == 0


def test_torch_optimization() -> None:
    module = agents.Perceptron(input_shape=(2,), output_size=1)
    torchopt = helpers.TorchOptimizer(module.parameters(), optimizerlib.OnePlusOne)
    # pylint: disable=not-callable
    x = module.forward(agents.torch.tensor(np.array([2, 3], dtype=np.float32))).item()
    torchopt.step(3.0)
    y = module.forward(agents.torch.tensor(np.array([2, 3], dtype=np.float32))).item()
    assert x != y
    # Other instanciations
    helpers.TorchOptimizer(module.parameters(), "OnePlusOne")
    with pytest.raises(TypeError):
        helpers.TorchOptimizer(module.parameters(), 12)  # type: ignore
