import numpy as np


class SupremeDynamicPrecisionOptimizerV2:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Initialize temperature and cooling parameters
        T = 1.1  # Starting temperature
        T_min = 0.001  # Minimum temperature threshold for annealing
        alpha = 0.92  # Cooling rate, fine-tuned for a more gradual decrease

        # Mutation and crossover parameters optimized further
        F = 0.75  # Mutation factor tuned for a better balance between exploration and exploitation
        CR = 0.85  # Crossover probability adjusted to maintain diversity while promoting good traits

        population_size = 80  # Adjusted population size for better performance within the budget
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Enhanced mutation dynamics and temperature-dependent acceptance
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Dynamic mutation factor influenced by temperature and progress
                dynamic_F = (
                    F * np.exp(-0.06 * T) * (0.7 + 0.3 * np.cos(2 * np.pi * evaluation_count / self.budget))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Improved acceptance criterion based on delta_fitness and temperature
                if delta_fitness < 0 or np.random.rand() < np.exp(-delta_fitness / T):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Adaptive cooling strategy that adjusts based on performance and remaining budget
            adaptive_cooling = alpha - 0.008 * np.sin(1.5 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
