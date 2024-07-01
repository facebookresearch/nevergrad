import numpy as np


class UltraRefinedDynamicPrecisionOptimizerV33:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Advanced temperature and cooling settings for optimal exploration and exploitation
        T = 1.2  # Increased initial temperature for broader initial exploration
        T_min = 0.0003  # Lower minimum temperature for sustained exploration at later stages
        alpha = 0.91  # Slower cooling rate to extend effective search period

        # Mutation and crossover parameters optimized for dynamic environments
        F = 0.78  # Adjusted Mutation factor for better exploration/exploitation balance
        CR = 0.88  # Enhanced Crossover probability to maintain sufficient genetic diversity

        population_size = 85  # Slightly increased population size for better sampling
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Implementing a dynamic mutation with a sigmoid-based adaptation for refined control
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Sigmoid-based dynamic mutation factor for more nuanced adaptation
                dynamic_F = (
                    F
                    * np.exp(-0.05 * T)
                    * (0.65 + 0.35 * np.tanh(4 * (evaluation_count / self.budget - 0.5)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Adaptive acceptance criteria incorporating a more intricate temperature function
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.065 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Adaptive cooling strategy with sinusoidal modulation for refined temperature control
            adaptive_cooling = alpha - 0.007 * np.sin(3.5 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
