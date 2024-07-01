import numpy as np


class PMFSA:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])

    def initialize(self):
        population_size = 100
        population = np.random.uniform(*self.bounds, (population_size, self.dimension))
        return population, population_size

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def adaptive_search(self, population, func):
        global_best_fitness = np.Inf
        global_best_individual = None

        evaluations = 0
        short_term_memory = []

        while evaluations < self.budget:
            fitness = self.evaluate(population, func)
            evaluations += len(population)

            # Update best solution globally
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < global_best_fitness:
                global_best_fitness = fitness[best_idx]
                global_best_individual = population[best_idx]

            # Multi-Level Feedback for population adaptation
            short_term_memory.append(fitness[best_idx])
            if len(short_term_memory) > 10:
                recent_trend = np.std(short_term_memory[-10:])
                if recent_trend < 0.05:
                    mutation_scale = 0.1
                else:
                    mutation_scale = 0.5
            else:
                mutation_scale = 0.3

            mutations = np.random.normal(0, mutation_scale, (len(population), self.dimension))
            population += mutations
            population = np.clip(population, *self.bounds)

            # Periodic reset with memory of best solutions
            if evaluations % 1000 == 0 and evaluations != 0:
                population = np.random.uniform(*self.bounds, (len(population) - 1, self.dimension))
                population = np.vstack([population, global_best_individual])

        return global_best_fitness, global_best_individual

    def __call__(self, func):
        initial_population, _ = self.initialize()
        best_fitness, best_solution = self.adaptive_search(initial_population, func)
        return best_fitness, best_solution
