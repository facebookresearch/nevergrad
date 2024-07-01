import numpy as np


class AdaptivePrecisionStrategicOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality of the problem
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Optimizer parameters
        population_size = 100
        mutation_factor = 0.8  # Adjusted mutation factor
        crossover_probability = 0.75  # Adjusted crossover probability
        elite_size = 5

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        # Adaptive mechanism for mutation and crossover
        success_rate = np.zeros(population_size)  # Track success for each individual

        while evaluations < self.budget:
            new_population = np.zeros_like(population)
            new_fitness = np.zeros_like(fitness)

            # Preserve elite solutions
            elite_indices = np.argsort(fitness)[:elite_size]
            new_population[:elite_size] = population[elite_indices]
            new_fitness[:elite_size] = fitness[elite_indices]

            for i in range(elite_size, population_size):
                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = population[indices]

                # Mutation with adaptive factor
                adaptive_mutation_factor = mutation_factor + 0.1 * (2 * success_rate[i] - 1)
                mutant = a + adaptive_mutation_factor * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Adaptive crossover probability
                adaptive_crossover_probability = crossover_probability + 0.1 * (2 * success_rate[i] - 1)

                # Crossover
                trial_vector = np.where(
                    np.random.rand(self.dim) < adaptive_crossover_probability, mutant, population[i]
                )
                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial_vector
                    new_fitness[i] = trial_fitness
                    success_rate[i] += 1  # Increment success count for this individual
                else:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]
                    success_rate[i] = max(0, success_rate[i] - 1)  # Decrement or maintain success rate

                if new_fitness[i] < best_fitness:
                    best_fitness = new_fitness[i]
                    best_solution = new_population[i]

            population = new_population
            fitness = new_fitness

        return best_fitness, best_solution
