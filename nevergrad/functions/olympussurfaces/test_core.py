# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from . import core


@pytest.mark.parametrize("kind", ["AckleyPath",
            "Dejong",
            "HyperEllipsoid",
            "Levy",
            "Michalewicz",
            "Rastrigin",
            "Rosenbrock",
            "Schwefel",
            "StyblinskiTang",
            "Zakharov",
            "DiscreteAckley",
            "DiscreteDoubleWell",
            "DiscreteMichalewicz",
            "LinearFunnel",
            "NarrowFunnel",
            "GaussianMixture"])
@pytest.mark.parametrize("noise_kind", ["GaussianNoise", "UniformNoise", "GammaNoise"])
def test_olympus_surface(kind: str, noise_kind: str) -> None:
    func = core.OlympusSurface(kind=kind, noise_kind=noise_kind)
    x = 2 * np.random.rand(func.dimension)
    value = func(x)  # should not touch boundaries, so value should be < np.inf
    assert isinstance(value, float)
    assert value < np.inf
