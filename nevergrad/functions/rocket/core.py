# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Based on https://raw.githubusercontent.com/purdue-orbital/rocket-simulation/master/Simulation2.py


import numpy as np
from nevergrad.parametrization import parameter as p
from ..base import ExperimentFunction
from .rocket import rocket as rocket

class Rocket(ExperimentFunction):

    def __init__(self) -> None:
        super().__init__(self._simulate_rocket, p.Array(shape=(24,)))
        self.register_initialization()

    def _simulate_rocket(self, x: np.ndarray) -> float:
        return rocket(x)

