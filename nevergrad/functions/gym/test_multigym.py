# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gym
import os
from unittest import SkipTest
import numpy as np
import pytest
from . import multigym


GYM_ENV_NAMES = multigym.GymMulti.get_env_names()


def test_multigym() -> None:
    raise SkipTest("Skipping GYM for now.")
    for env_name in multigym.GymMulti.ng_gym:
        assert env_name in GYM_ENV_NAMES, f"{env_name} unknown!"
        assert env_name not in multigym.NO_LENGTH, f"{env_name} in no length and in ng_gym!"
    for env_name in multigym.GUARANTEED_GYM_ENV_NAMES:
        assert env_name in GYM_ENV_NAMES, f"{env_name} should be guaranteed!"
    assert len(GYM_ENV_NAMES) >= 10 or os.name == "nt"


# def test_cartpole() -> None:
#    func = multigym.GymMulti(name="CartPole-v0", control="neural", neural_factor=1, randomized=True)
#    candidate = func.parametrization.sample()
#    results = [func.evaluation_function(candidate) for _ in range(40)]
#    assert min(results) != max(results), "CartPole should not be deterministic."
#
#
# def test_sparse_cartpole() -> None:
#    func = multigym.GymMulti(
#        name="CartPole-v0", control="neural", neural_factor=1, randomized=True, sparse_limit=2
#    )
#    param = func.parametrization.sample()
#    func(*param.args, **param.kwargs)
#    candidate = func.parametrization.sample()
#    results = [func.evaluation_function(candidate) for _ in range(40)]
#    assert min(results) != max(results), "CartPole should not be deterministic."
#
#
@pytest.mark.parametrize("name", ["LunarLander-v2"])  # type: ignore
def test_run_multigym(name: str) -> None:
    raise SkipTest("Skipping GYM for now.")
    if os.name == "nt" or np.random.randint(8) or "CubeCrash" in name:
        raise SkipTest("Skipping Windows and running only 1 out of 8")
    if "ANM" in name:
        raise SkipTest("We skip ANM6Easy and related problems.")

    func = multigym.GymMulti(randomized=False, neural_factor=None)
    x = np.zeros(func.dimension)
    value = func(x)
    np.testing.assert_almost_equal(value, 178.2, decimal=2)
    i = GYM_ENV_NAMES.index(name)
    control = multigym.CONTROLLERS[i % len(multigym.CONTROLLERS)]
    print(f"Working with {control} on {name}.")
    func = multigym.GymMulti(
        name,
        control,
        neural_factor=(None if (control == "linear" or "conformant" in control) else 1),
        randomized=bool(np.random.randint(2)),
    )
    y = func.parametrization.sample()
    func(y.value)
    if "stac" in control and "Acrobat" in name:  # Let's check if the memory works.
        np.testing.assert_almost_equal(func(y.value), 500, decimal=2)
    if "stac" in control and "Pendulum-v0" in name:  # Let's check if the memory works.
        np.testing.assert_almost_equal(func(y.value), 1720.39, decimal=2)


gym.envs.register(
    id="TupleActionSpace-v0", entry_point="nevergrad.functions.gym:TupleActionSpace", max_episode_steps=168
)


# def test_tuple_action_space_random() -> None:
#    func = multigym.GymMulti(name="TupleActionSpace-v0", control="conformant", neural_factor=None)
#    val = ng.optimizers.DiscreteOnePlusOne(func.parametrization, budget=100).minimize(func).value
#    reward = min(func(func.parametrization.sample().value) for _ in range(3))
#    assert reward > -80000  # type: ignore
#    assert reward < 0  # type: ignore
#    assert func(val) < reward  # type: ignore


def test_tuple_action_space_neural() -> None:
    raise SkipTest("Skipping GYM for now.")
    func = multigym.GymMulti(name="TupleActionSpace-v0", control="neural", neural_factor=1)
    results_neural = [func(np.random.normal(size=func.dimension)) for _ in range(10)]
    assert min(results_neural) != max(results_neural)  # type: ignore
    assert all(int(r) == r for r in results_neural)  # type: ignore
