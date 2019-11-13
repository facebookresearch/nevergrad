import nevergrad as ng
import numpy as np
import random
import math

# Number of stocks (dams).
N = 3
pi = math.pi

# Real life is more complicated!

class Agent():
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
         





all_agents = []

year_to_day_ratio = 2.  # Ratio between variation of consumption in the year and variation of consumption in the day
constant_to_year_ratio = 0
back_to_normal = 0.03  # How much of the gap with normal is cancelled at each iteration.
consumption_noise = 1.
num_thermal_plants = 7
average_consumption = constant_to_year_ratio * year_to_day_ratio
thermal_power_capacity = [c * average_consumption for c in list(np.random.rand(num_thermal_plants))]
thermal_power_prices = [c for c in list(np.random.rand(num_thermal_plants))]

for i in range(N):
    all_agents += [Agent(6 + N + 2*num_thermal_plants, 1)]

def simulate_power_system(input_x):
    x = [r for r in input_x]
    for a in all_agents:
        assert(len(x) >= a.GetParamNumbers())
        a.SetParams(x[:a.GetParamNumbers()])
        x = x[a.GetParamNumbers():]
    assert(len(x) == 0)

    # Assume empty initial stocks.
    stocks = [0.] * N   
    # Nonsense delays.
    delay = [np.cos(i) for i in range(N)]
    cost = 0
    # Loop on time steps.
    consumption=0.
    for t in range(365*24):

        # Rain
        for i in range(N):
            stocks[i] += 0.5*(1.+np.cos(2*pi*t/(24*365) + delay[i])) * np.random.rand()
        # Consumption model.
        base_consumption = (constant_to_year_ratio*year_to_day_ratio*1. 
                +0.5*year_to_day_ratio*(1.+np.cos(2*pi*t/(24*365))) + 0.5*(1.+np.cos(2*pi*t/24)))
        consumption = max(0., consumption + consumption_noise*np.random.rand() + back_to_normal * (base_consumption - consumption))
        # Water usage.
        needed = consumption

        # Setting inputs for all agents.
        base_x = [needed, average_consumption, year_to_day_ratio, constant_to_year_ratio, back_to_normal, consumption_noise]
        x = base_x + thermal_power_capacity + thermal_power_prices + stocks

        # Prices as a decomposition tool!
        price = [a.GetOutput(x)[0][0] for a in all_agents]
        volume = [s for s in stocks]
        price += thermal_power_prices
        volume += thermal_power_capacity

        agents = [all_agents[i] for i in sorted(range(N), key=lambda i: price[i])]
        for i in range(N):
            if needed == 0:
                break
            production = min(volume[i], needed)
            cost += production * price[i]
    return cost

# Simple instrumentation: just the number of params.
instrumentation = sum([a.GetParamNumbers() for a in all_agents])
optimizer = ng.optimizers.OnePlusOne(instrumentation=instrumentation, budget=500)

losses = []
for i in range(500):
    candidate = optimizer.ask()
    v = simulate_power_system(candidate.data)
    losses += [v]
    optimizer.tell(candidate, v)

import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel('iteration number')
plt.ylabel('cost')
plt.show()
import time
time.sleep(30)

recommendation = optimizer.recommend()  #minimize(square)
print(recommendation)  # optimal args and kwargs







