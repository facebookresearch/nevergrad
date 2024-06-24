import numpy as np


class ARISA:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])

    def initialize(self):
        population_size = 50
        population = np.random.uniform(*self.bounds, (population_size, self.dimension))
        return population, population_size

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def adaptive_search(self, population, func):
        global_best_fitness = np.Inf
        global_best_individual = None

        evaluations = 0
        history_changes = []

        while evaluations < self.budget:
            fitness = self.evaluate(population, func)
            evaluations += len(population)

            # Update best solution
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < global_best_fitness:
                global_best_fitness = fitness[best_idx]
                global_best_individual = population[best_idx]

            # Adaptive mutation based on historical performance
            if history_changes:
                mutation_scale = 0.1 if np.std(history_changes) < 0.1 else 0.5
            else:
                mutation_scale = 0.3
            mutations = np.random.normal(0, mutation_scale, (len(population), self.dimension))
            population = population + mutations
            population = np.clip(population, *self.bounds)

            # Record and adjust
            current_best_fitness = fitness[best_idx]
            history_changes.append(current_best_fitness)

            # Thresholding for dynamic adaptation
            if len(history_changes) > 5:
                if np.std(history_changes[-5:]) < 0.01 * np.abs(history_changes[-1]):
                    population = np.random.uniform(*self.bounds, (len(population), self.dimension))

        return global_best_fitness, global_best_individual

    def __call__(self, func):
        initial_population, _ = self.initialize()
        best_fitness, best_solution = self.adaptive_search(initial_population, func)
        return best_fitness, best_solution
