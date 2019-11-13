import nevergrad as ng
import numpy as np
import random
import math
import math.pi as pi

# Number of stocks (dams).
N = 3
year_to_day_ratio = 2.  # Ratio between variation of consumption in the year and variation of consumption in the day
constant_to_year_ratio = 0
back_to_normal = 0.03  # How much of the gap with normal is cancelled at each iteration.
consumption_noise = 1.

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
        self.layers += [np.random.rand(layer_width, output_width)]
        assert len(self.layers) == layers

    def GetParamNumbers(self):
        return sum([prod(l.shape) for l in self.layers])

    def SetParams(self, w):
        assert(len(w) == self.GetParamNumbers)
        for l in self.layers:
            l = np.reshape(w, l.shape)

    def GetOutput(self, inp):
        output = inp
        for l in [self.layers[i] for i in range(len(self.layers) - 1)]:
            output = np.tanh(output * l)
        return output * self.layers[-1]
         





all_agents = []
for i in range(N):
    all_agents += Agent(80, 1)

average_consumption = constant_to_year_ratio * year_to_day_ratio
power_volumes = [c * average_consumption for c in list(np.random.rand(7))]
power_price = [c for c in list(np.random.rand(7))]

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
    delay = [cos(i) for i in range(N)]
    cost = 0
    # Loop on time steps.
    for t in range(365*24):

        # Rain
        for i in range(N):
            stocks[i] += 0.5*(1.+maths.cos(2*pi*t/(24*365) + delay[i])) * np.random.rand()
        # Consumption model.
        base_consumption = (constant_to_year_ratio*year_to_day_ratio*1. 
                +0.5*year_to_day_ratio*(1.+math.cos(2*pi*t/(24*365))) + 0.5*(1.+math.cos(2*pi*t/24)))
        consumption = max(0., consumption + consumption_noise*np.random.rand() + back_to_normal * (base_consumption - consumption))
        # Water usage.
        needed = consumption

        # Prices as a decomposition tool!
        price = [a.GetOutput() for a in all_agents]
        volume = [s for s in stocks]
        price += power_prices
        volume += power_volumes

        agents = [all_agents[i] for i in sorted(range(N), key=lambda i: price[i])]
        for i in range(N):
            if needed == 0:
                break
            production = min(volume[i], needed)
            cost += production * price[i]

    return cost

optimizer = ng.optimizers.OnePlusOne(instrumentation=sum(a.GetParamNumbers() for a in all_agents), budget=500)

for i in range(500):
    candidate = optimizer.ask()
    v = simulate_power_system(candidate.data)
    optimizer.tell(candidate, v)

recommendation = optimizer.recommend()  #minimize(square)
print(recommendation)  # optimal args and kwargs







