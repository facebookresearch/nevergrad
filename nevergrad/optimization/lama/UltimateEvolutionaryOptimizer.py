import numpy as np


class UltimateEvolutionaryOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5
        self.lb = -5.0  # Lower bound of the search space
        self.ub = 5.0  # Upper bound of the search space

    def __call__(self, func):
        # Initialize temperature with a slightly higher starting point and refine minimum threshold
        T = 1.2  # Starting temperature increased for more aggressive initial exploration
        T_min = 0.0003  # Lower temperature threshold for fine-tuned exploitation
        alpha = 0.95  # Slightly slower cooling rate to allow more thorough search at each temperature level

        # Mutation and crossover parameters optimized for a balance between diversification and intensification
        F = 0.8  # Higher mutation factor for more effective exploration early in the process
        CR = 0.85  # Lowered crossover probability to ensure better offspring quality

        population_size = 100  # Increased population size for more diverse initial solutions
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

                # Dynamic mutation factor incorporating more complex adaptive behavior
                dynamic_F = (
                    F * np.exp(-0.1 * T) * (0.7 + 0.3 * np.cos(2 * np.pi * evaluation_count / self.budget))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Superior acceptance criterion that adapts better to changes in fitness landscape
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.06 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Non-linear adaptive cooling with an added modulation to account for search stagnation
            adaptive_cooling = alpha - 0.015 * np.sin(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
