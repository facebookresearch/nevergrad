import numpy as np


class RefinedGradientBoostedOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality of the problem
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Optimizer parameters
        population_size = 100
        mutation_factor = 0.5  # Lower mutation factor to start with finer mutations
        crossover_rate = 0.9  # High crossover rate for stronger exploitation
        grad_step_size = 0.01  # Step size for gradient approximation
        adaptive_rate = 0.02  # More conservative adaptive rate

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        # Using a hybrid gradient and evolutionary strategy
        while evaluations < self.budget:
            for i in range(population_size):
                # Gradient mutation and differential mutation combined
                grad_mutant = population[i] + np.random.randn(self.dim) * grad_step_size
                grad_mutant = np.clip(grad_mutant, self.lower_bound, self.upper_bound)
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                diff_mutant = a + mutation_factor * (b - c)
                diff_mutant = np.clip(diff_mutant, self.lower_bound, self.upper_bound)

                # Construct trial vector using both mutations
                trial_vector = np.where(np.random.rand(self.dim) < 0.5, grad_mutant, diff_mutant)
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < crossover_rate
                trial_vector = np.where(crossover_mask, trial_vector, population[i])

                trial_fitness = func(trial_vector)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial_vector

            # Adaptive updates for mutation and crossover rates
            mutation_factor = max(0.1, mutation_factor - adaptive_rate * np.random.randn())
            crossover_rate = min(1.0, max(0.5, crossover_rate + adaptive_rate * np.random.randn()))

        return best_fitness, best_solution
