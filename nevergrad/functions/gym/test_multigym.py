# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from unittest import SkipTest
import numpy as np
import pytest
from . import multigym


GYM_ENV_NAMES = multigym.GymMulti.get_env_names()


def test_multigym() -> None:
    for env_name in multigym.GymMulti.ng_gym:
        assert env_name in GYM_ENV_NAMES, f"{env_name} unknown!"
        assert env_name not in multigym.NO_LENGTH, f"{env_name} in no length and in ng_gym!"
    for env_name in multigym.GUARANTEED_GYM_ENV_NAMES:
        assert env_name in GYM_ENV_NAMES, f"{env_name} should be guaranteed!"
    assert len(GYM_ENV_NAMES) >= 16 or os.name == "nt"


def test_compiler_gym() -> None:
    func = multigym.CompilerGym(17)
    candidate = func.parametrization.sample()
    results = [func.evaluation_function(candidate) for _ in range(4)]
    assert min(results) == max(results), "CompilerGym should be deterministic."


def test_roulette() -> None:
    func = multigym.GymMulti(name="CartPole-v0", control="neural", neural_factor=1, randomized=True)
    results = [func(np.zeros(func.dimension)) for _ in range(40)]
    assert min(results) != max(results), "CartPole should not be deterministic."
    candidate = func.parametrization.sample()
    results = [func.evaluation_function(candidate) for _ in range(40)]
    assert min(results) != max(results), "CartPole should not be deterministic."


@pytest.mark.parametrize("name", GYM_ENV_NAMES)  # type: ignore
def test_run_multigym(name: str) -> None:
    if os.name == "nt" or np.random.randint(8) or "CubeCrash" in name:
        raise SkipTest("Skipping Windows and running only 1 out of 8")
    func = multigym.GymMulti(randomized=False, neural_factor=None)
    x = np.zeros(func.dimension)
    value = func(x)
    np.testing.assert_almost_equal(value, 184.07, decimal=2)
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
        np.testing.assert_almost_equal(func(y.value), 1688.82, decimal=2)
