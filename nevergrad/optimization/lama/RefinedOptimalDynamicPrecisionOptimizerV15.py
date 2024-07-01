import numpy as np


class RefinedOptimalDynamicPrecisionOptimizerV15:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is consistently set to 5
        self.lb = -5.0  # Lower boundary of the search space
        self.ub = 5.0  # Upper boundary of the search space

    def __call__(self, func):
        # Initialize temperature and cooling parameters with slight refinements
        T = 1.1  # Slightly reduced starting temperature for a more balanced exploration
        T_min = 0.0004  # Optimized minimum temperature for deeper late-stage precision
        alpha = 0.91  # Adjusted cooling rate for more gradual transition in phases

        # Mutation and crossover parameters are finely tuned for optimal performance
        F = 0.78  # Fine-tuned Mutation factor
        CR = 0.88  # Fine-tuned Crossover probability

        population_size = 82  # Optimized population size for efficient search
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Dynamic mutation with sigmoid-based adaptation for responsive mutation control
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                # Dynamic mutation factor using an improved sigmoid for more responsive control
                dynamic_F = (
                    F
                    * np.exp(-0.065 * T)
                    * (0.65 + 0.35 * np.tanh(3.5 * (evaluation_count / self.budget - 0.5)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Refined acceptance criteria with adjusted sensitive temperature function
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.07 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Upgraded adaptive cooling strategy with sinusoidal modulation
            adaptive_cooling = alpha - 0.0075 * np.cos(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
