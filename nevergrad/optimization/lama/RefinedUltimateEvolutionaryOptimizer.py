import numpy as np


class RefinedUltimateEvolutionaryOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5
        self.lb = -5.0  # Lower bound of the search space
        self.ub = 5.0  # Upper bound of the search space

    def __call__(self, func):
        # Initialize temperature with a refined starting value and minimum threshold
        T = 1.1  # Slightly decreased starting temperature for optimal initial exploration
        T_min = 0.0005  # Lower minimum temperature threshold for fine-tuned exploitation
        alpha = 0.92  # Modified cooling rate to allow extended search at each temperature level

        # Mutation and crossover parameters optimized based on previous results
        F = 0.75  # Adjusted mutation factor to help balance exploration and exploitation
        CR = 0.88  # Adjusted crossover probability to maintain diversity while improving offspring quality

        population_size = 80  # Slightly increased population size for more diverse initial solutions
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Enhanced mutation dynamics with a focus on adaptive mutation factors
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                # Dynamic mutation factor incorporating improved adaptive behavior
                dynamic_F = (
                    F
                    * np.exp(-0.12 * T)
                    * (0.65 + 0.35 * np.tanh(2 * np.pi * evaluation_count / self.budget))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Advanced acceptance criterion adapting better to changes in fitness landscape
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.07 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Non-linear adaptive cooling with an updated modulation to account for search efficiency
            adaptive_cooling = alpha - 0.012 * np.sin(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
