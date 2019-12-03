# run this with:
# echo 'import nevergrad.examples.powersystem' | python 
# or (e.g. MacOS):
# echo 'import nevergrad.examples.powersystem' | pythonw 

from nevergrad.functions.powersystems.core import PowerSystem
from nevergrad.optimization import optimizerlib

width = 6
depth = 6
for num_dams in [3,6,12]:
    for year_to_day_ratio in [0.5, 2., 8.]:
        for back_to_normal in [0.03, 0.1, 0.5]:
            for num_thermal_plants in [3, 6, 12]:
                for constant_to_year_ratio in [0.25, 1., 4.]:
#     year_to_day_ratio: float = 2.,  # Ratio between std of consumption in the year and std of consumption in the day.
#     constant_to_year_ratio: float = 1.,  # Ratio between constant baseline consumption and std of consumption in the year.
#     back_to_normal: float = 0.5,  # Part of the variability which is forgotten at each time step.
#     consumption_noise: float = 0.1,  # Instantaneous variability.
#     num_thermal_plants: int = 7,  # Number of thermal plants.
#     num_years: int = 1,  # Number of years.
#     failure_cost: float = 500.,  # Cost of not satisfying the demand. Equivalent to an expensive infinite capacity thermal plant.

                    power_system_loss = PowerSystem(num_dams=num_dams, depth=depth, width=width, year_to_day_ratio=year_to_day_ratio, back_to_normal=back_to_normal, num_thermal_plants=num_thermal_plants, constant_to_year_ratio=constant_to_year_ratio) 
                    optimizer = optimizerlib.SplitOptimizer9(instrumentation=power_system_loss.dimension, budget=500, num_workers=10) 
                    optimizer.minimize(power_system_loss)
                    power_system_loss(optimizer.provide_recommendation().data)
                    power_system_loss.make_plots(f"ps_{num_dams}dams_{depth}_{width}_ytdr{year_to_day_ratio}_btn{back_to_normal}_num_thermal_plants{num_thermal_plants}_ctyr{constant_to_year_ratio}.png")




