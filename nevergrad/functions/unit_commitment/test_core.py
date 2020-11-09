# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import nevergrad as ng
from . import core

def test_unit_commitment_p1() -> None:
    T = 10
    N = 5
    func = core.UnitCommitmentProblem1(T_points=T, N_generators=N)
    x = np.random.rand(N, T)
    u = np.random.rand(N, T)
    value = func.function(x=x, u=u)
    optimizer = ng.optimizers.NGO(parametrization=func.parametrization, budget=100)
    recommendation = optimizer.minimize(func.function)
    assert value < np.inf
    assert func(**recommendation.kwargs) < np.inf
