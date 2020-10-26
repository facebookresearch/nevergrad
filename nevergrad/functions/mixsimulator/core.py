# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Based on https://github.com/Foloso/MixSimulator/tree/nevergrad_experiment

import numpy as np
from ..base import ExperimentFunction
from mixsimulator.MixSimulator import MixSimulator  #type: ignore


class OptimizeMix(ExperimentFunction):
    """
        MixSimulator is an application with an optimization model for calculating 
        and simulating the least cost of an energy mix under certain constraints.
        
        For now, it uses a default dataset (more will be added soon).
        
        For more information, visit : https://github.com/Foloso/MixSimulator     
        
        Parameters
        ----------
        time: int
            total time over which it evaluates the mix (must in hour)
                
    """
    def __init__(self,time: int = 168) -> None:
        
        parameters = self._parametrization(time = time)
        super().__init__(self._simulate_mix , parameters)
        
        self.register_initialization(time=time)
        self.add_descriptors(time=time)

    def _simulate_mix(self, x: np.ndarray) -> float:
        mix = MixSimulator()
        mix.set_data_to("Toamasina")
        mix.set_penalisation_cost(100)
        mix.set_carbon_cost(10)
        return mix.loss_function(x)
        
    def _parametrization(self,time : int = 168):
        mix = MixSimulator()
        mix.set_data_to("Toamasina")
        
        #If time == 168h (one_week) --> dim = 672
        params = mix.get_opt_params(time)
        params.set_name("dims")
        
        return params
