import numpy as np


class AdvancedBalancedExplorationOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population_size = 80
        mutation_factor = 0.8
        crossover_probability = 0.7
        elite_size = 5

        # Initialize population and evaluate
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        # Adaptation factors
        adaptive_mutation = np.full(population_size, mutation_factor)
        adaptive_crossover = np.full(population_size, crossover_probability)
        success_tracker = np.zeros(population_size)

        while evaluations < self.budget:
            new_population = np.zeros_like(population)
            new_fitness = np.zeros_like(fitness)

            # Elite retention
            elite_indices = np.argsort(fitness)[:elite_size]
            new_population[:elite_size] = population[elite_indices]
            new_fitness[:elite_size] = fitness[elite_indices]

            for i in range(elite_size, population_size):
                # Mutation and Crossover
                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = population[indices]

                # Mutation with adaptive factors
                mutant = a + adaptive_mutation[i] * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                trial = np.where(np.random.rand(self.dim) < adaptive_crossover[i], mutant, population[i])
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
                    success_tracker[i] = max(0, success_tracker[i] - 1)

                # Update adaptive mutation and crossover probabilities
                if success_tracker[i] > 2:
                    adaptive_mutation[i] = min(1.0, adaptive_mutation[i] + 0.05)
                    adaptive_crossover[i] = min(1.0, adaptive_crossover[i] + 0.05)
                elif success_tracker[i] == 0:
                    adaptive_mutation[i] = max(0.5, adaptive_mutation[i] - 0.05)
                    adaptive_crossover[i] = max(0.5, adaptive_crossover[i] - 0.05)

                # Update best solution
                if new_fitness[i] < best_fitness:
                    best_fitness = new_fitness[i]
                    best_solution = new_population[i]

            population = new_population
            fitness = new_fitness

        return best_fitness, best_solution
