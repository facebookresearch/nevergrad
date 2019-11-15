# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This code is based on a code and ideas by Emmanuel Centeno and Antoine Moreau,
# University Clermont Auvergne, CNRS, SIGMA Clermont, Institut Pascal

import copy
from math import sqrt, tan, pi
from typing import Any
from typing import List
import numpy as np
from nevergrad.common.typetools import ArrayLike
from ... import instrumentation as inst
from ...instrumentation.core import Variable
from ...instrumentation.multivariables import Instrumentation
pi = np.pi

#optimizer = ng.optimizers.OnePlusOne(instrumentation=instrumentation, budget=500)
#
#losses = []
#for i in range(optim_steps):
#    candidate = optimizer.ask()
#    v, hydro_prod, hydro_prod_per_ts, consumption_per_ts = simulate_power_system(candidate.data)
#    losses += [min(v, min([float("Inf")] + losses))]
#    optimizer.tell(candidate, v)
#
#ax = plt.subplot(1, 2, 1)
#ax.set_xlabel('iteration number')
#ax.plot(losses, label='losses') 
#ax = plt.subplot(1, 2, 2)
#ax.plot(np.linspace(0,1,len(consumption_per_ts)), consumption_per_ts, label='consumption')
#ax.plot(np.linspace(0,1,len(hydro_prod_per_ts)), hydro_prod_per_ts, label='hydro')
#for i in range(N):
#    hydro_ts = [hydro_prod_per_ts[j] for j in range(i, len(hydro_prod_per_ts), N)]
#    ax.plot(np.linspace(0,1,len(hydro_ts)), hydro_ts, label='dam ' + str(i) + ' prod')
#ax.set_xlabel('time step')
#ax.set_ylabel('production per ts')
#ax.legend() #(l1, l2))
#plt.show()
#plt.savefig("ps.png")
#plt.waitforbuttonpress()
#recommendation = optimizer.recommend()  #minimize(square)
#print(recommendation)  # optimal args and kwargs





# Real life is more complicated! This is a very simple model.
class Agent():
    """An agent has an input size, an output size, a number of layers, a width of its internal layers 
    (a.k.a number of neurons per hidden layer)."""
    def __init__(self, input_size, output_size, layers=2, layer_width=20):
        assert(layers >= 2)
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []
        self.layers += [np.random.rand(input_size, layer_width)]
        for i in range(layers-2):
            self.layers += [np.random.rand(layer_width, layer_width)]
        self.layers += [np.random.rand(layer_width, output_size)]
        assert len(self.layers) == layers

    def GetParamNumbers(self):
        return sum([np.prod(l.shape) for l in self.layers])

    def SetParams(self, ww):
        w = [w for w in ww]
        assert(len(w) == self.GetParamNumbers())
        for l in self.layers:
            s = np.prod(l.shape)
            l = np.reshape(np.array(w[:s]), l.shape)
            w = w[s:]

    def GetOutput(self, inp):
        output = np.array(inp).reshape(1, len(inp))
        for l in [self.layers[i] for i in range(len(self.layers) - 1)]:
            output = np.tanh(np.matmul(output, l))
        return np.matmul(output, self.layers[-1])




class PowerSystemVariable(Variable):

    def __init__(self, num_stocks:int, depth: int, width: int) -> None:  #, dimension: int) -> None:
        
        # Simple instrumentation: just the number of params.
        super().__init__()
        #self.num_stocks = num_stocks
        #self.depth = depth
        #self.width = width
        #self._specs.update(dimension=dimension)

    # pylint: disable=unused-argument
    def _data_to_arguments(self, data: np.ndarray, deterministic: bool = True) -> Any:
        #assert len(data) == self.dimension
        return (data), {}

    def __repr__(self) -> str:
        return "PowerSystems" #+ str(self.num_stocks) + "stocks_" + str(self.depth) + "layers_" + str(self.width) + "neurons"


class PowerSystem(inst.InstrumentedFunction):
    """
    Parameters
    ----------
    num_stocks: number of stocks to be managed
    depth: number of layers in the neural networks
    width: number of neurons per hidden layer
    """

    def _simulate_power_system(self, input_x: np.ndarray):
        x = list(input_x)
        dam_managers = self.dam_managers
        for a in dam_managers:
            assert(len(x) >= a.GetParamNumbers())
            a.SetParams(np.array(x[:a.GetParamNumbers()]))
            x = x[a.GetParamNumbers():]
        assert(len(x) == 0)
        N = self.N
        # Assume empty initial stocks.
        stocks = [0.] * N   
        # Nonsense delays.
        delay = [np.cos(i) for i in range(N)]
        cost = 0.
        # Loop on time steps.
        consumption = 0.
        hydro_prod_per_time_step: List[float] = []
        consumption_per_time_step: List[float] = []
        for t in range(365*24*self.number_of_years):
    
            # Rain
            for i in range(N):
                stocks[i] += 0.5*(1.+np.cos(2*pi*t/(24*365) + delay[i])) * np.random.rand()
            # Consumption model.
            base_consumption = (self.constant_to_year_ratio*self.year_to_day_ratio*1. 
                    +0.5*self.year_to_day_ratio*(1.+np.cos(2*pi*t/(24*365))) + 0.5*(1.+np.cos(2*pi*t/24)))
            consumption = max(0., consumption + self.consumption_noise*np.random.rand() + self.back_to_normal * (base_consumption - consumption))
            consumption_per_time_step += [consumption]
            # Water usage.
            needed = consumption
    
            # Setting inputs for all agents.
            base_x = [np.cos(2*pi*t/24.), np.sin(2*pi*t/24.), np.cos(2*pi*t/(365*24)), np.sin(2*pi*t/(365*24)), needed, self.average_consumption, self.year_to_day_ratio, self.constant_to_year_ratio, self.back_to_normal, self.consumption_noise]
            x = list(base_x + self.thermal_power_capacity + self.thermal_power_prices + stocks)
    
            # Prices as a decomposition tool!
            price: List[float] = [a.GetOutput(np.array(x))[0][0] for a in dam_managers]
            volume: List[float] = [s for s in stocks]
            dam_index: List[int] = list(range(len(price)))
            price += self.thermal_power_prices
            volume += self.thermal_power_capacity
            dam_index += [-1] * len(price)
            
            assert(len(price) == N + self.num_thermal_plants)
            hydro_prod: List[float] = [0.] * N
            for i in range(len(price)):
                if needed == 0:
                    break
                production = min(volume[i], needed)
                if dam_index[i] >= 0:
                    hydro_prod[dam_index[i]] += production
                else:
                    cost += production * price[i]
                needed -= production
            cost += 500. * needed
            hydro_prod_per_time_step += hydro_prod
        return cost  # Other data of interest: , hydro_prod, hydro_prod_per_time_step, consumption_per_time_step
        #super().__init__(self._simulate_power_system, PowerSystemVariable(num_stocks, depth, width)) 

    def __init__(self, num_stocks: int = 3, depth: int = 3, width: int = 3) -> None:
        dam_managers: List[Any] = []
        # Number of stocks (dams).
        self.N = num_stocks
        N = self.N
        # Parameters describing the problem.
        self.year_to_day_ratio = 2. * N  # Ratio between variation of consumption in the year and variation of consumption in the day
        self.constant_to_year_ratio = N * 2.
        self.back_to_normal = 0.03  # How much of the gap with normal is cancelled at each iteration.
        self.consumption_noise = 1.
        self.num_thermal_plants = 7
        self.number_of_years = 1
        
        self.average_consumption = self.constant_to_year_ratio * self.year_to_day_ratio
        self.thermal_power_capacity = [c * self.average_consumption for c in list(np.random.rand(self.num_thermal_plants))]
        self.thermal_power_prices = [c for c in list(np.random.rand(self.num_thermal_plants))]
        for i in range(N):
            dam_managers += [Agent(10 + N + 2*self.num_thermal_plants, 1)]
        dimension = sum([a.GetParamNumbers() for a in dam_managers])
        self.dam_managers = dam_managers
        super().__init__(self._simulate_power_system, Instrumentation(inst.var.Array(dimension)))
        self._descriptors.update(num_stocks=num_stocks, depth=depth, width=width)

