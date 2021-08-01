# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# This code is based on a code and ideas by Emmanuel Centeno and Antoine Moreau,
# University Clermont Auvergne, CNRS, SIGMA Clermont, Institut Pascal

import typing as tp
from math import pi, cos, sin
import matplotlib.pyplot as plt
import numpy as np
from nevergrad.parametrization import parameter as p
from ..base import ExperimentFunction


class Agent:
    """An agent has an input size, an output size, a number of layers, a width of its internal layers
    (a.k.a number of neurons per hidden layer)."""

    def __init__(self, input_size: int, output_size: int, layers: int = 3, layer_width: int = 14) -> None:
        assert layers >= 2
        self.input_size = input_size
        self.output_size = output_size
        self.layers = [np.zeros((layer_width, input_size))]
        for _ in range(layers - 2):
            self.layers += [np.zeros((layer_width, layer_width))]
        self.layers += [np.zeros((output_size, layer_width))]
        assert len(self.layers) == layers

    @property
    def dimension(self) -> int:
        return sum(layer.size for layer in self.layers)

    def set_parameters(self, weights: np.ndarray) -> None:
        if weights.size != self.dimension:
            raise ValueError(f"length = {weights.size} instead of {self.dimension}: {weights}.")
        start = 0
        for i, layer in enumerate(self.layers):
            numel = layer.size
            self.layers[i] = weights[start : start + numel].reshape(layer.shape)
            start += numel
        if start != weights.size:
            raise RuntimeError("Unexpected runtime error when distributing the weights")

    def get_output(self, data: np.ndarray) -> np.ndarray:
        for l in self.layers[:-1]:
            data = np.tanh(l @ data)
        return self.layers[-1] @ data  # type: ignore


# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-statements,too-many-locals
class PowerSystem(ExperimentFunction):
    """Very simple model of a power system.
    Real life is more complicated!

    Parameters
    ----------
    num_dams: int
        number of dams to be managed
    depth: int
        number of layers in the neural networks
    width: int
        number of neurons per hidden layer
    year_to_day_ratio: float = 2.
        Ratio between std of consumption in the year and std of consumption in the day.
    constant_to_year_ratio: float
        Ratio between constant baseline consumption and std of consumption in the year.
    back_to_normal: float
        Part of the variability which is forgotten at each time step.
    consumption_noise: float
        Instantaneous variability.
    num_thermal_plants: int
        Number of thermal plants.
    num_years: float
        Number of years.
    failure_cost: float
        Cost of not satisfying the demand. Equivalent to an expensive infinite capacity thermal plant.
    """

    def __init__(
        self,
        num_dams: int = 13,
        depth: int = 3,
        width: int = 3,
        year_to_day_ratio: float = 2.0,
        constant_to_year_ratio: float = 1.0,
        back_to_normal: float = 0.5,
        consumption_noise: float = 0.1,
        num_thermal_plants: int = 7,
        num_years: float = 1.0,
        failure_cost: float = 500.0,
    ) -> None:
        self.num_dams = num_dams
        self.losses: tp.List[float] = []
        self.marginal_costs: tp.List[float] = []
        # Parameters describing the problem.
        self.year_to_day_ratio = year_to_day_ratio
        self.constant_to_year_ratio = constant_to_year_ratio
        self.back_to_normal = back_to_normal
        self.consumption_noise = consumption_noise
        self.num_thermal_plants = num_thermal_plants
        self.number_of_years = num_years
        self.failure_cost = failure_cost
        self.hydro_prod_per_time_step: tp.List[tp.Any] = []  # TODO @oteytaud initial values?
        self.consumption_per_time_step: tp.List[tp.Any] = []

        self.average_consumption = self.constant_to_year_ratio * self.year_to_day_ratio
        self.thermal_power_capacity = self.average_consumption * np.random.rand(self.num_thermal_plants)
        self.thermal_power_prices = np.random.rand(num_thermal_plants)
        self.dam_agents = [
            Agent(10 + num_dams + 2 * self.num_thermal_plants, depth, width) for _ in range(num_dams)
        ]
        parameter = p.Instrumentation(
            *[p.Array(shape=(int(a.dimension),)) for a in self.dam_agents]
        ).set_name("")
        super().__init__(self._simulate_power_system, parameter)
        self.parametrization.function.deterministic = False

    def get_num_vars(self) -> tp.List[tp.Any]:
        return [m.dimension for m in self.dam_agents]

    def _simulate_power_system(self, *arrays: np.ndarray) -> float:
        failure_cost = (
            self.failure_cost
        )  # Cost of power demand which is not satisfied (equivalent to a expensive infinite thermal group).
        dam_agents = self.dam_agents
        for agent, array in zip(dam_agents, arrays):
            agent.set_parameters(array)

        self.marginal_costs = []

        num_dams = int(self.num_dams)
        # Assume empty initial stocks.
        stocks = np.zeros((num_dams,))
        # Nonsense delays.
        delay = np.cos(np.arange(num_dams))
        cost = 0.0
        # Loop on time steps.
        num_time_steps = int(365 * 24 * self.number_of_years)
        consumption = 0.0
        hydro_prod_per_time_step: tp.List[tp.Any] = []
        consumption_per_time_step: tp.List[float] = []
        for t in range(num_time_steps):

            # Rain
            stocks += 0.5 * (1.0 + np.cos(2 * pi * t / (24 * 365) + delay)) * np.random.rand(num_dams)
            # Consumption model.
            base_consumption = (
                self.constant_to_year_ratio * self.year_to_day_ratio
                + 0.5 * self.year_to_day_ratio * (1.0 + cos(2 * pi * t / (24 * 365)))
                + 0.5 * (1.0 + cos(2 * pi * t / 24))
            )

            if t == 0:
                consumption = base_consumption
            else:
                consumption = max(
                    0.0,
                    consumption
                    + self.consumption_noise * (np.random.rand() - 0.5)
                    + self.back_to_normal * (base_consumption - consumption),
                )
            consumption_per_time_step += [consumption]
            # "Needed" stores what we need, and will decrease as we use various power plants for producing.
            needed = consumption

            # Setting inputs for all agents.
            base_x = [
                cos(2 * pi * t / 24.0),
                sin(2 * pi * t / 24.0),
                cos(2 * pi * t / (365 * 24)),
                sin(2 * pi * t / (365 * 24)),
                needed,
                self.average_consumption,
                self.year_to_day_ratio,
                self.constant_to_year_ratio,
                self.back_to_normal,
                self.consumption_noise,
            ]

            x = np.concatenate((base_x, self.thermal_power_capacity, self.thermal_power_prices, stocks))

            # Prices as a decomposition tool!
            price: np.ndarray = np.asarray([a.get_output(x)[0] for a in dam_agents])
            dam_index = np.arange(num_dams)
            price = np.concatenate((price, self.thermal_power_prices))
            capacity = np.concatenate((stocks, self.thermal_power_capacity))
            dam_index = np.concatenate((dam_index, -1 * np.ones(len(price), dtype=int)))

            assert len(price) == num_dams + self.num_thermal_plants
            hydro_prod: np.ndarray = np.zeros(num_dams)

            # Let us rank power plants by production cost.
            order = sorted(range(len(price)), key=lambda x: price[x])  # pylint: disable=cell-var-from-loop
            price = price[order]
            capacity = capacity[order]
            dam_index = dam_index[order]

            # Using power plants in their cost order, so that we use cheap power plants first.
            marginal_cost = 0.0
            for i, _ in enumerate(price):
                if needed <= 0:
                    break
                production = min(capacity[i], needed)
                # If this is a dam, producing will reduce the stock.
                if dam_index[i] >= 0:
                    hydro_prod[dam_index[i]] += production  # Let us log the hydro prod for this dam.
                    stocks[dam_index[i]] -= production
                    assert stocks[dam_index[i]] >= -1e-7
                else:
                    # If this is not a dam, we pay for using thermal plants.
                    cost += production * price[i]
                    if production > 1e-7:
                        marginal_cost = price[i]

                needed -= production
            # Cost in case of failures -- this is
            # harming industries and hospitals, so it can be penalized.
            cost += failure_cost * needed
            if needed > 1e-7:
                marginal_cost = failure_cost
            self.marginal_costs += [marginal_cost]

            hydro_prod_per_time_step += [hydro_prod]
        # Other data of interest: , hydro_prod, hydro_prod_per_time_step, consumption_per_time_step
        assert len(hydro_prod_per_time_step) == num_time_steps  # Each time steps has 1 value per dam.
        assert len(consumption_per_time_step) == num_time_steps
        self.hydro_prod_per_time_step = hydro_prod_per_time_step
        self.consumption_per_time_step = consumption_per_time_step
        self.losses += [cost]
        return cost

    def make_plots(self, filename: str = "ps.png") -> None:
        losses = self.losses
        num_dams = self.num_dams
        consumption_per_ts = self.consumption_per_time_step
        hydro_prod_per_ts = self.hydro_prod_per_time_step
        total_hydro_prod_per_ts = [sum(h) for h in hydro_prod_per_ts]

        # Utility function for plotting per year or per day.
        def block(x: tp.List[float]) -> tp.List[float]:
            result: tp.List[float] = []
            step = int(np.sqrt(len(x)))
            for i in range(0, len(x), step):
                result += [sum(x[i : i + step]) / len(x[i : i + step])]
            return result

        def block24(x: tp.List[float]) -> tp.List[float]:
            result: tp.List[float] = []
            for i in range(0, len(x), 24):
                result += [sum(x[i : i + 24]) / len(x[i : i + 24])]
            if len(x) != len(result) * 24:
                print(len(x), len(result) * 24)
            return result

        def deblock24(x: tp.List[float]) -> tp.List[float]:
            result = [0.0] * 24
            for i, _ in enumerate(x):
                result[i % 24] += x[i] / 24.0
            return result

        plt.clf()
        # Plot the optimization run.
        ax = plt.subplot(2, 2, 1)
        ax.set_xlabel("iteration number")
        smoothed_losses = block(losses)
        ax.plot(np.linspace(0, 1, len(losses)), losses, label="losses")
        ax.plot(np.linspace(0, 1, len(smoothed_losses)), smoothed_losses, label="smoothed losses")
        ax.legend(loc="best")

        # Plotting marginal costs.
        ax = plt.subplot(2, 2, 4)
        marginal_cost_per_hour = deblock24(self.marginal_costs)
        marginal_cost_per_day = block24(self.marginal_costs)
        ax.plot(
            np.linspace(0, 0.5, len(marginal_cost_per_hour)),
            marginal_cost_per_hour,
            label="marginal cost per hour",
        )
        ax.plot(
            np.linspace(0.5, 1, len(marginal_cost_per_day)),
            marginal_cost_per_day,
            label="marginal cost per day",
        )
        ax.legend(loc="best")

        # Plot consumption per day and decomposition of production.
        ax = plt.subplot(2, 2, 2)
        consumption_per_day = block24(consumption_per_ts)
        hydro_prod_per_day = block24(total_hydro_prod_per_ts)
        ax.plot(np.linspace(1, 365, len(consumption_per_day)), consumption_per_day, label="consumption")
        ax.plot(np.linspace(1, 365, len(hydro_prod_per_day)), hydro_prod_per_day, label="hydro")
        for i in range(min(num_dams, 3)):
            hydro_ts = [h[i] for h in hydro_prod_per_ts]

            hydro_day = block24(hydro_ts)
            ax.plot(np.linspace(1, 365, len(hydro_day)), hydro_day, label="dam " + str(i) + " prod")
        ax.set_xlabel("time step")
        ax.set_ylabel("production per day")
        ax.legend(loc="best")

        # Plot consumption per hour of the day and decomposition of production.
        ax = plt.subplot(2, 2, 3)
        consumption_per_hour = deblock24(consumption_per_ts)
        hydro_prod_per_hour = deblock24(total_hydro_prod_per_ts)
        ax.plot(np.linspace(1, 24, len(consumption_per_hour)), consumption_per_hour, label="consumption")
        ax.plot(np.linspace(1, 24, len(hydro_prod_per_hour)), hydro_prod_per_hour, label="hydro")
        for i in range(min(num_dams, 3)):
            hydro_ts = [h[i] for h in hydro_prod_per_ts]
            hydro_hour = deblock24(hydro_ts)
            ax.plot(np.linspace(1, 24, len(hydro_hour)), hydro_hour, label="dam " + str(i) + " prod")
        ax.set_xlabel("time step")
        ax.set_ylabel("production per hour")
        ax.legend(loc="best")
        plt.savefig(filename)
