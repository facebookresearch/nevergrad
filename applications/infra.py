# E.g. run with
# export PYTHONPATH=${PYTHONPATH}:.
# python applications/infra.py
# or (macos):
# pythonw applications/infra.py

import matplotlib.pyplot as plt
import nevergrad as ng
import numpy as np
import random
import math
pi = math.pi

# We consider a power systems problem.
# We have N stocks.
# Each stock is managed by an agent.
# Agents collaborate for minimizing the total cost.

# Number of stocks (dams).
N = 7

# Optimization budget.
optim_steps = 8

# Real life is more complicated! This is a very simple model.

class Agent():
    """An agent has an input size, an output size, a number of layers, a width of its internal layers 
    (a.k.a number of neurons per hidden layer)."""
    def __init__(self, input_size, output_size, layers=2, layer_width=20):
        assert(layers >= 2)
        self.input_size = input_size
        self.output_size = output_size
        self.layers = []
        self.layers += [np.random.rand(int(input_size), int(layer_width))]
        for i in range(layers-2):
            self.layers += [np.random.rand(int(layer_width), int(layer_width))]
        self.layers += [np.random.rand(int(layer_width), int(output_size))]
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
         





dam_managers = []

# Parameters describing the problem.
year_to_day_ratio = 1.  # Ratio between variation of consumption in the year and variation of consumption in the day
constant_to_year_ratio = 1.
dam_output_factor = 1.0
back_to_normal = 0.5  # How much of the gap with normal is cancelled at each iteration.
consumption_noise = 0.1
num_thermal_plants = 7
number_of_years = 1

average_consumption = constant_to_year_ratio * year_to_day_ratio
thermal_power_capacity = [c * average_consumption for c in list(np.random.rand(num_thermal_plants))]
thermal_power_prices = [c for c in list(np.random.rand(num_thermal_plants))]

for i in range(N):
    dam_managers += [Agent(10 + N + 2*num_thermal_plants, 1)]

def simulate_power_system(input_x):
    x = [r for r in input_x]
    for a in dam_managers:
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
    hydro_prod_per_time_step_and_dam = []
    consumption_per_time_step = []
    hydro_prod_per_ts = []
    for t in range(365*24*number_of_years):

        # Rain
        for i in range(N):
            stocks[i] += 0.5*dam_output_factor*(1.+np.cos(2*pi*t/(24*365) + delay[i])) * np.random.rand()
        # Consumption model.
        base_consumption = (constant_to_year_ratio*year_to_day_ratio  +0.5*year_to_day_ratio*(1.+np.cos(2*pi*t/(24*365))) + 0.5*(1.+np.cos(2*pi*t/24)))
        if t == 0:
            consumption = base_consumption
        else:
            consumption = max(0., consumption + consumption_noise*(np.random.rand()-0.5) + back_to_normal * (base_consumption - consumption))
        consumption_per_time_step += [consumption]
        # Water usage.
        needed = consumption

        # Setting inputs for all agents.
        base_x = [np.cos(2*pi*t/24.), np.sin(2*pi*t/24.), np.cos(2*pi*t/(365.*24)), np.sin(2*pi*t/(365.*24)), needed, average_consumption, year_to_day_ratio, constant_to_year_ratio, back_to_normal, consumption_noise]
        x = base_x + thermal_power_capacity + thermal_power_prices + stocks

        # Prices as a decomposition tool!
        price = [a.GetOutput(x)[0][0] for a in dam_managers]
        volume = [s for s in stocks]
        dam_index = list(range(len(price)))
        price += thermal_power_prices
        volume += thermal_power_capacity
        dam_index += [-1] * len(price)
        assert(len(price) == N + num_thermal_plants)
        hydro_prod = [0.] * N
        hydro_prod_per_ts += [0.]
        for i in range(len(price)):
            if needed == 0:
                break
            production = min(volume[i], needed)
            if dam_index[i] >= 0:
                stocks[dam_index[i]] -= production
                assert stocks[dam_index[i]] >= -1e-7
                hydro_prod[dam_index[i]] += production
                hydro_prod_per_ts[-1] += production
            else:
                cost += production * price[i]
            needed -= production
        cost += 500. * needed
        hydro_prod_per_time_step_and_dam += hydro_prod
    assert len(hydro_prod_per_ts) == number_of_years * 365*24
    assert len(hydro_prod_per_time_step_and_dam) == number_of_years * 365*24 * N
    assert len(consumption_per_time_step) == number_of_years * 24 * 365
    return cost, hydro_prod_per_ts, hydro_prod_per_time_step_and_dam, consumption_per_time_step

# Simple instrumentation: just the number of params.
instrumentation = sum([a.GetParamNumbers() for a in dam_managers])
optimizer = ng.optimizers.OnePlusOne(instrumentation=instrumentation, budget=500)

losses = []
for i in range(optim_steps):
    candidate = optimizer.ask()
    v, hydro_prod_per_ts, hydro_prod_per_ts_per_dam, consumption_per_ts = simulate_power_system(candidate.data)
    losses += [min(v, min([float("Inf")] + losses))]
    optimizer.tell(candidate, v)

# Plot the optimization run.
ax = plt.subplot(2, 2, 1)
ax.set_xlabel('iteration number')
ax.plot(losses, label='losses') 
ax.legend(loc='best')

# Plot consumption per day and decomposition of production.
ax = plt.subplot(2, 2, 2)
def block24(x):
    assert len(x) % 24 == 0
    result = []
    for i in range(0, len(x), 24):
        result += [sum(x[i:i+24])]
    if len(x) != len(result) * 24:
        print("failure:", len(x), len(result))
    assert len(x) == len(result) * 24
    return result
consumption_per_day = block24(consumption_per_ts)
assert np.abs(sum(consumption_per_day) - sum(consumption_per_ts)) < 0.0001
assert np.abs(sum(hydro_prod_per_ts_per_dam) - sum(hydro_prod_per_ts)) < 0.0001
print(len(hydro_prod_per_ts), ' should be divided by 24')
hydro_prod_per_day = block24(hydro_prod_per_ts)
assert np.abs(sum(hydro_prod_per_ts_per_dam) - sum(hydro_prod_per_day)) < 0.0001
assert len(hydro_prod_per_day) == len(consumption_per_day)
print('daily hydro prod=', hydro_prod_per_day[:5])
print('daily consumption=', consumption_per_day[:5])
ax.plot(np.linspace(1,365,len(consumption_per_day)), consumption_per_day, label='consumption')
if True:
    ax.plot(np.linspace(1,365,len(hydro_prod_per_day)), hydro_prod_per_day, label='hydro')
    assert sum(consumption_per_day) >= sum(hydro_prod_per_ts_per_dam)
    assert len(hydro_prod_per_ts_per_dam) % N == 0
    assert len(hydro_prod_per_ts_per_dam) % (N * 24) == 0
    assert len(hydro_prod_per_ts_per_dam) % (N * 24 * 365) == 0
    assert abs(sum(hydro_prod_per_ts_per_dam) - sum(hydro_prod_per_ts)) < 1e-4
    assert sum(hydro_prod_per_ts) <= sum(consumption_per_ts)
    #hydro_ts = [0.] * (len(hydro_prod_per_ts_per_dam) // N)
    #for j in range(len(hydro_prod_per_ts_per_dam)):
    #    hydro_ts[j % N] += hydro_prod_per_ts_per_dam[j]
    for i in range(min(N, 3)):
        # Let us consider a specific dam
        hydro_ts = [hydro_prod_per_ts_per_dam[j] for j
                in range(i, len(hydro_prod_per_ts_per_dam), N)]
        hydro_day = block24(hydro_ts)
        ax.plot(np.linspace(1,365,len(hydro_day)), hydro_day, label='dam ' + str(i) + ' prod')
ax.set_xlabel('time step')
ax.set_ylabel('production per day')
ax.legend(loc='best')

# Plot consumption per hour of the day and decomposition of production.
ax = plt.subplot(2, 2, 3)
def deblock24(x):
    result = [0] * 24
    for i in range(0, len(x)):
        result[i % 24] += x[i]
    assert len(result) == 24
    return result
        
consumption_per_hour = deblock24(consumption_per_ts)
print(consumption_per_hour)
hydro_prod_per_hour = deblock24(hydro_prod_per_ts)
print(hydro_prod_per_hour)
assert len(consumption_per_hour) == 24
ax.plot(np.linspace(1,24,len(consumption_per_hour)), consumption_per_hour, label='consumption')
ax.plot(np.linspace(1,24,len(hydro_prod_per_hour)), hydro_prod_per_hour, label='hydro')
ax.plot(np.linspace(1,24,len(hydro_prod_per_hour)), [a-b for a, b in zip(consumption_per_hour, hydro_prod_per_hour)], label='thermal')
for i in range(min(N,3)):
    hydro_ts = [hydro_prod_per_ts_per_dam[j] for j in range(i, len(hydro_prod_per_ts), N)]
    hydro_hour = deblock24(hydro_ts)
    ax.plot(np.linspace(1,24,len(hydro_hour)), hydro_hour, label='dam ' + str(i) + ' prod')
ax.set_xlabel('time step')
ax.set_ylabel('production per hour')
ax.legend(loc='best')
plt.savefig("ps.png")
plt.show()
plt.waitforbuttonpress()
recommendation = optimizer.recommend()  #minimize(square)
print(recommendation)  # optimal args and kwargs







