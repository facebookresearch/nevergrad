import numpy as np


class UltraRefinedDynamicPrecisionOptimizerV40:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Initial temperature with further refined controls for late-stage optimization
        T = 1.1
        T_min = 0.0001
        alpha = 0.93  # More gradual cooling to provide a deeper search in later stages

        # Mutation and crossover parameters optimized for more aggressive exploration and exploitation
        F = 0.77
        CR = 0.88  # Slightly increased for enhanced mixing of genetic information

        population_size = 85  # Slightly larger population for more diverse genetic material
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Dynamic mutation factor using logistic function for more precise adjustments
                dynamic_F = F * (
                    0.75 + 0.25 * (1 / (1 + np.exp(-10 * (evaluation_count / self.budget - 0.5))))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1

                # Enhanced fitness-based acceptance criteria
                delta_fitness = trial_fitness - fitness[i]
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.05 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Adaptive cooling with a more complex modulation pattern for temperature
            T *= alpha - 0.007 * np.sin(3 * np.pi * evaluation_count / self.budget)

        return f_opt, x_opt
