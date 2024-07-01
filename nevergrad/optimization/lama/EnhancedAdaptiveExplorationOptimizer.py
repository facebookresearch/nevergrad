import numpy as np


class EnhancedAdaptiveExplorationOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Search space dimension
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population_size = 250
        mutation_factor = 0.8
        crossover_rate = 0.9
        elite_size = 25

        # Initialize population and evaluate
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        # Adaptive mechanisms
        success_tracker = np.zeros(population_size)
        mutation_factors = np.full(population_size, mutation_factor)
        crossover_rates = np.full(population_size, crossover_rate)

        while evaluations < self.budget:
            new_population = np.zeros_like(population)
            new_fitness = np.zeros_like(fitness)

            # Elite retention
            elite_indices = np.argsort(fitness)[:elite_size]
            new_population[:elite_size] = population[elite_indices]
            new_fitness[:elite_size] = fitness[elite_indices]

            for i in range(elite_size, population_size):
                # Parents selection
                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = population[indices]

                # Mutation with adaptive mutation rate
                mutant = a + mutation_factors[i] * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover with adaptive crossover rate
                trial = np.where(np.random.rand(self.dim) < crossover_rates[i], mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    new_fitness[i] = trial_fitness
                    success_tracker[i] += 1
                else:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]

                # Adapt mutation and crossover rates based on success
                if success_tracker[i] >= 5:
                    mutation_factors[i] = min(1.0, mutation_factors[i] + 0.02)
                    crossover_rates[i] = min(1.0, crossover_rates[i] + 0.05)
                elif success_tracker[i] == 0:
                    mutation_factors[i] = max(0.1, mutation_factors[i] - 0.02)
                    crossover_rates[i] = max(0.1, crossover_rates[i] - 0.05)

                # Update best solution
                if new_fitness[i] < best_fitness:
                    best_fitness = new_fitness[i]
                    best_solution = new_population[i]

            # Refresh success tracker periodically to avoid stagnation
            if evaluations % 1000 == 0:
                success_tracker = np.zeros(population_size)

            population = new_population
            fitness = new_fitness

        return best_fitness, best_solution
