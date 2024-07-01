import numpy as np


class UltraRefinedDynamicPrecisionOptimizerV27:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Further refined temperature and cooling parameters
        T = 1.2  # Slightly higher initial temperature for more aggressive early exploration
        T_min = 0.0003  # Lower minimum temperature to enable deeper late-stage exploration
        alpha = 0.9  # Adjusted cooling rate to better balance exploration and exploitation

        # Mutation and crossover parameters are optimized
        F = 0.78  # Refined Mutation factor for optimal exploration-exploitation balance
        CR = 0.88  # Adjusted Crossover probability to enhance genetic diversity

        population_size = 85  # Population size fine-tuned to enhance convergence rates
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Implement a dynamic mutation approach with enhanced sigmoid-based modulation
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Adjusted dynamic mutation factor incorporates precision control
                dynamic_F = (
                    F * np.exp(-0.08 * T) * (0.8 + 0.2 * np.tanh(4 * (evaluation_count / self.budget - 0.5)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Enhanced acceptance criteria with modified temperature dependency
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.07 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Modified adaptive cooling strategy using sinusoidal modulation
            adaptive_cooling = alpha - 0.009 * np.sin(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
