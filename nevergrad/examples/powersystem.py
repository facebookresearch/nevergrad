# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# run this with:
# echo 'import nevergrad.examples.powersystem' | python
# or (e.g. MacOS):
# echo 'import nevergrad.examples.powersystem' | pythonw

import nevergrad as ng
from nevergrad.functions.powersystems.core import PowerSystem

budget = 3500
width = 6
depth = 6
num_dams = 6
year_to_day_ratio = 0.5
back_to_normal = 0.5
num_thermal_plants = 6
constant_to_year_ratio = 4.0
# Default values for various options:
#     year_to_day_ratio: float = 2.,  # Ratio between std of consumption in the year and std of consumption in the day.
#     constant_to_year_ratio: float = 1.,  # Ratio between constant baseline consumption and std of consumption in the year.
#     back_to_normal: float = 0.5,  # Part of the variability which is forgotten at each time step.
#     consumption_noise: float = 0.1,  # Instantaneous variability.
#     num_thermal_plants: int = 7,  # Number of thermal plants.
#     num_years: int = 1,  # Number of years.
#     failure_cost: float = 500.,  # Cost of not satisfying the demand. Equivalent to an expensive infinite capacity thermal plant.

power_system_loss = PowerSystem(
    num_dams=num_dams,
    depth=depth,
    width=width,
    year_to_day_ratio=year_to_day_ratio,
    back_to_normal=back_to_normal,
    num_thermal_plants=num_thermal_plants,
    constant_to_year_ratio=constant_to_year_ratio,
)
optimizer = ng.optimizers.SplitOptimizer(
    parametrization=power_system_loss.dimension, budget=budget, num_workers=10
)
optimizer.minimize(power_system_loss)
power_system_loss(optimizer.provide_recommendation().value)
power_system_loss.make_plots(
    f"ps_{num_dams}dams_{depth}_{width}_ytdr{year_to_day_ratio}_btn{back_to_normal}"
    f"_num_thermal_plants{num_thermal_plants}_ctyr{constant_to_year_ratio}_budget{budget}.png"
)
