import numpy as np


class PrecisionBalancedOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality of the problem
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Optimizer parameters
        population_size = 150
        mutation_factor = 0.5  # Lower mutation factor to increase precision in exploitation
        crossover_rate = 0.8  # Higher crossover to explore beneficial traits
        learning_rate = 0.1  # Learning rate for adaptive mechanisms

        # Initial population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        evaluations = population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        # Evolution process
        while evaluations < self.budget:
            for i in range(population_size):
                # Mutation using a differential evolution strategy
                indices = [index for index in range(population_size) if index != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant_vector = a + mutation_factor * (b - c)
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, population[i])

                trial_fitness = func(trial_vector)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

                    # Update best solution found
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial_vector

            # Adaptive mechanism to fine-tune parameters
            mutation_factor = max(0.1, mutation_factor - learning_rate * np.random.randn())
            crossover_rate = min(1.0, max(0.5, crossover_rate + learning_rate * np.random.randn()))

        return best_fitness, best_solution
