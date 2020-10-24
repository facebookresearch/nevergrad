# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Based on https://github.com/Foloso/MixSimulator/tree/nevergrad_experiment

import numpy as np
#from nevergrad.parametrization import parameter as p
from ..base import ExperimentFunction
from mixsimulator.MixSimulator import MixSimulator  #type: ignore


class OptimizeMix(ExperimentFunction):

    def __init__(self) -> None:
        mix = MixSimulator()
        mix.set_data_to("Toamasina")
        #If time == one_week --> dim = 672
        super().__init__(self._simulate_mix, mix.get_opt_params(672))
        #Rename parametrization
        super().parametrization.set_name("dim672")
        self.register_initialization()

    def _simulate_mix(self, x: np.ndarray) -> float:
        mix = MixSimulator()
        mix.set_data_to("Toamasina")
        mix.set_penalisation_cost(100)
        mix.set_carbon_cost(10)
        return mix.loss_func(x)
