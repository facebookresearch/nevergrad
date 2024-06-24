import numpy as np


class UltraRefinedDynamicPrecisionOptimizerV39:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Enhanced initial temperature for more explorative early stages
        T = 1.2
        T_min = 0.0003  # Reduced minimum temperature for deeper late-stage exploration
        alpha = 0.91  # Further slowed cooling rate

        # Fine-tuned mutation and crossover parameters
        F = 0.78  # Adjusted Mutation factor
        CR = 0.9  # Higher Crossover probability to enhance diversity

        population_size = 82  # Optimized population size
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Dynamic mutation strategy with an enhanced sigmoid-based adaptation
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                dynamic_F = F * (0.75 + 0.25 * np.sin(np.pi * evaluation_count / self.budget))
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1

                # Utilizing a more dynamic temperature-dependent acceptance probability
                if trial_fitness < fitness[i] or np.random.rand() < np.exp(
                    -(trial_fitness - fitness[i])
                    / (T * (1 + 0.05 * np.tanh(3 * (evaluation_count / self.budget - 0.5))))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Adaptive cooling increased complexity with phase-based modulation
            T *= alpha - 0.005 * np.cos(2 * np.pi * evaluation_count / self.budget)

        return f_opt, x_opt
