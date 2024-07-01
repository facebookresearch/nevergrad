import numpy as np


class UltraRefinedDynamicPrecisionOptimizerV54:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Initialize temperature and cooling parameters
        T = 1.20  # Adjusted higher initial Temperature for aggressive exploration at the start
        T_min = 0.001  # Lower final temperature for finer optimization at later stages
        alpha = 0.95  # Higher cooling rate to sustain a longer exploration phase with more gradual cooling

        # Mutation and crossover parameters are finely tuned
        F = 0.78  # Adjusted Mutation factor to maintain a better balance between exploration and exploitation
        CR = 0.85  # Crossover probability adjusted to ensure robust mixing of features

        population_size = 90  # Increased population size to provide more diversity
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Dynamic mutation approach with logistic growth rate adaptation
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Adjusting dynamic mutation factor using a logistic model for better control over exploration
                dynamic_F = F / (1 + np.exp(-5 * (evaluation_count / self.budget - 0.5))) * (b - c)
                mutant = np.clip(a + dynamic_F, self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Incorporating a temperature-dependent acceptance probability
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.07 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Adaptive cooling strategy with a polynomial decay model
            adaptive_cooling = alpha - 0.007 * (evaluation_count / self.budget) ** 2
            T *= adaptive_cooling

        return f_opt, x_opt
