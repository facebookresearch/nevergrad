# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Based on https://github.com/Foloso/MixSimulator/tree/nevergrad_experiment

from mixsimulator.MixSimulator import MixSimulator  #type: ignore
from ..base import ExperimentFunction


class OptimizeMix(ExperimentFunction):
    """
        MixSimulator is an application with an optimization model for calculating 
        and simulating the least cost of an energy mix under certain constraints.
        
        For now, it uses a default dataset (more will be added soon).
        
        For more information, visit : https://github.com/Foloso/MixSimulator     
        
        Parameters
        ----------
        time: int
            total time over which it evaluates the mix (must be in hour)
                
    """
    def __init__(self, time: int = 168) -> None:
        self._mix = MixSimulator()
        self._mix.set_data_to("Toamasina")
        self._mix.set_penalisation_cost(100)
        self._mix.set_carbon_cost(10)
        parameters = self._mix.get_opt_params(time)
        parameters.set_name("dims")
        super().__init__(self._mix.loss_function, parameters)
        self.register_initialization(time=time)
        self.add_descriptors(time=time)
    
