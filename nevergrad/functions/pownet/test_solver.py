# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from . import solver

def test_solver_artificial() -> None:
    solver.pownet_solver(location="artificial")
