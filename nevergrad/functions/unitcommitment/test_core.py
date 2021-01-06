# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from . import core


def test_unit_commitment_p1() -> None:
    np.random.seed(0)
    T = 10
    N = 5
    func = core.UnitCommitmentProblem(problem_name="semi-continuous", num_timepoints=T, num_generators=N)
    op_out = np.ones((N, T))
    op_states = np.ones((N, T))
    value = func.function(operational_output=op_out, operational_states=op_states)
    assert np.allclose([value], [38721960.61493097], rtol=1e-04, atol=1e-05)
