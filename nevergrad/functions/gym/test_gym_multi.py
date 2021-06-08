# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pytest
import os
from . import gym_multi


def test_gym_multi() -> None:
    for env_name in gym_multi.GymMulti.ng_gym:
        assert env_name in gym_multi.GymMulti.env_names, f"{env_name} unknown!"
        assert env_name not in gym_multi.NO_LENGTH, f"{env_name} in no length and in ng_gym!"
    for env_name in gym_multi.GUARANTEED_GYM_ENV_NAMES:
        assert env_name in gym_multi.GymMulti.env_names, f"{env_name} should be guaranteed!"
    assert len(gym_multi.GYM_ENV_NAMES) >= 26 or os.name == "nt"


def test_compiler_gym() -> None:
    func = gym_multi.CompilerGym(17)
    candidate = func.parametrization.sample()
    results = [func.evaluation_function(candidate) for _ in range(4)]
    assert min(results) == max(results), "CompilerGym should be deterministic."


def test_roulette() -> None:
    print(gym_multi.GymMulti.ng_gym)
    func = gym_multi.GymMulti(name="Roulette-v0", control="neural", scaling_factor=1, randomized=True)
    results = [func(np.zeros(func.dimension)) for _ in range(40)]
    assert min(results) != max(results), "Roulette should not be deterministic."
    candidate = func.parametrization.sample()
    results = [func.evaluation_function(candidate) for _ in range(40)]
    assert min(results) != max(results), "Roulette should not be deterministic."


@pytest.mark.parametrize("name", gym_multi.GYM_ENV_NAMES)
def test_run_gym_multi(name) -> None:
    if os.name != "nt" and all(np.random.randint(2, size=3, dtype=bool)):
        func = gym_multi.GymMulti(randomized=False)
        x = np.zeros(func.dimension)
        value = func(x)
        np.testing.assert_almost_equal(value, 93.35, decimal=2)
        i = gym_multi.GYM_ENV_NAMES.index(name)
        control = gym_multi.CONTROLLERS[i % len(gym_multi.CONTROLLERS)]
        print(f"Working with {control} on {name}.")
        func = gym_multi.GymMulti(
            name,
            control,
            randomized=bool(np.random.randint(2)),
        )
        y = func.parametrization.sample()
        func(y.value)
        if "stac" in control:  # Let's check if the memory works.
            func(y.value)
