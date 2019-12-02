from nevergrad.functions.powersystems.core import PowerSystem
from nevergrad.optimization import optimizerlib

num_dams = 3
width = 3
depth = 3
#     year_to_day_ratio: float = 2.,  # Ratio between std of consumption in the year and std of consumption in the day.
#     constant_to_year_ratio: float = 1.,  # Ratio between constant baseline consumption and std of consumption in the year.
#     back_to_normal: float = 0.5,  # Part of the variability which is forgotten at each time step.
#     consumption_noise: float = 0.1,  # Instantaneous variability.
#     num_thermal_plants: int = 7,  # Number of thermal plants.
#     num_years: int = 1,  # Number of years.
#     failure_cost: float = 500.,  # Cost of not satisfying the demand. Equivalent to an expensive infinite capacity thermal plant.

power_system_loss = PowerSystem(num_dams=3, depth=3, width=3)

optimizer = optimizerlib.SplitOptimizer3(instrumentation=power_system_loss.dimension, budget=200, num_workers=10)

optimizer.minimize(power_system_loss)

power_system_loss(optimizer.provide_recommendation().data)
power_system_loss.make_plots(f"ps_{num_dams}dams_{depth}_{width}")




