import numpy as np


class EliteRefinedAdaptivePrecisionOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Initialize temperature and cooling parameters
        T = 1.2  # Initial higher temperature for more aggressive global exploration
        T_min = 0.0008  # Lower minimum temperature for deep exploitation
        alpha = 0.95  # Slower cooling rate to extend the exploration phase

        # Optimized mutation and crossover parameters for a balance between diversity and convergence
        F_base = 0.85  # Higher base mutation factor to encourage diverse mutations
        CR_base = 0.88  # Adjusted crossover probability for optimal diversity in offspring

        population_size = 90  # Increased population size for enhanced search capabilities
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Evolution loop with dynamic mutation and crossover influenced by temperature and solution quality
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Dynamic mutation factor adjusted by an exponential decay based on temperature and search progress
                dynamic_F = (
                    F_base
                    * np.exp(-0.15 * T)
                    * (0.7 + 0.3 * np.cos(1.5 * np.pi * evaluation_count / self.budget))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                CR_dynamic = CR_base - 0.12 * np.sin(2 * np.pi * evaluation_count / self.budget)
                cross_points = np.random.rand(self.dim) < CR_dynamic
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Enhanced adaptive acceptance criterion based on delta_fitness and temperature
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.08 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Adjusted cooling strategy with a progressive rate based on performance and remaining budget
            adaptive_cooling = alpha - 0.02 * np.cos(1.8 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
