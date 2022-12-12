# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import nevergrad as ng
from . import core
import pytest


@pytest.mark.parametrize("kind", core.OlympusSurface.SURFACE_KINDS)
@pytest.mark.parametrize("noise_kind", ["GaussianNoise", "UniformNoise", "GammaNoise"])
def test_olympus_surface(kind: str, noise_kind: str) -> None:
    try:
        func = core.OlympusSurface(kind=kind, noise_kind=noise_kind)
    except Exception as e:
        if os.name == "nt":
            raise ng.errors.UnsupportedExperiment("Unavailable under Windows.")
        else:
            raise e
    func2 = core.OlympusSurface(kind=kind, noise_kind=noise_kind)  # Let us check the randomization.
    x = 2 * np.random.rand(func.dimension)
    value = func(x)  # should not touch boundaries, so value should be < np.inf
    value2 = func2(x)  # should not touch boundaries, so value should be < np.inf
    assert isinstance(value, float)
    assert value < np.inf
    assert value != value2 or noise_kind == "GammaNoise"


# @pytest.mark.parametrize("dataset_kind", core.OlympusEmulator.DATASETS)
# @pytest.mark.parametrize("model_kind", ["NeuralNet"])  # ["BayesNeuralNet", "NeuralNet"])
# def test_olympus_emulator(dataset_kind: str, model_kind: str) -> None:
#    func = core.OlympusEmulator(dataset_kind=dataset_kind, model_kind=model_kind)
#    x = 2 * np.random.rand(func.dimension)
#    value = func(x)  # should not touch boundaries, so value should be < np.inf
#    assert isinstance(value, float)
#    assert value < np.inf
