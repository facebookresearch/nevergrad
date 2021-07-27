# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
from unittest import SkipTest
from . import core
import pytest


@pytest.mark.parametrize(
    "kind",
    core.OlympusSurface.get_surfaces_kinds(),
)
@pytest.mark.parametrize("noise_kind", ["GaussianNoise", "UniformNoise", "GammaNoise"])
def test_olympus_surface(kind: str, noise_kind: str) -> None:
    if os.name == "nt":
        raise SkipTest("Skipping Windows and running only 1 out of 8")
    func = core.OlympusSurface(kind=kind, noise_kind=noise_kind)
    func2 = core.OlympusSurface(kind=kind, noise_kind=noise_kind)  # Let us check the randomization.
    x = 2 * np.random.rand(func.dimension)
    value = func(x)  # should not touch boundaries, so value should be < np.inf
    value2 = func2(x)  # should not touch boundaries, so value should be < np.inf
    assert isinstance(value, float)
    assert value < np.inf
    assert value != value2


@pytest.mark.parametrize("dataset_kind", core.OlympusEmulator.get_datasets())
@pytest.mark.parametrize("model_kind", ["BayesNeuralNet","NeuralNet"])
def test_olympus_emulator(dataset_kind: str, model_kind: str) -> None:
    func = core.OlympusEmulator(dataset_kind=dataset_kind, model_kind=model_kind)
    x = 2 * np.random.rand(func.dimension)
    value = func(x)  # should not touch boundaries, so value should be < np.inf
    assert isinstance(value, float)
    assert value < np.inf
