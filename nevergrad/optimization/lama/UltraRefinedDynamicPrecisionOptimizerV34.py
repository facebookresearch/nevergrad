import numpy as np


class UltraRefinedDynamicPrecisionOptimizerV34:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Initialize temperature and cooling parameters with refined settings
        T = 1.18  # Modified starting temperature for broader initial exploration
        T_min = 0.0003  # Further lowered minimum temperature for deeper late-stage exploration
        alpha = 0.90  # Adjusted cooling rate for a slower reduction in temperature

        # Mutation and crossover parameters finely tuned for dynamic response
        F = 0.77  # Adjusted Mutation factor for optimal balance between exploration and exploitation
        CR = 0.89  # Higher Crossover probability to ensure better genetic diversity and mix

        population_size = 90  # Moderately increased population size for more robust sampling
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Integrate a more responsive dynamic mutation with a nuanced sigmoid adaptation
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Improved dynamic mutation factor for more balanced search adaptation
                dynamic_F = (
                    F * np.exp(-0.06 * T) * (0.6 + 0.4 * np.tanh(4 * (evaluation_count / self.budget - 0.5)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1

                if trial_fitness < fitness[i] or np.random.rand() < np.exp(-(trial_fitness - fitness[i]) / T):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Advanced adaptive cooling strategy with sinusoidal modulation for precise control
            adaptive_cooling = alpha - 0.006 * np.sin(4 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
