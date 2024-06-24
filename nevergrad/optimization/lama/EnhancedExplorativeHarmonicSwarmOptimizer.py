import numpy as np


class EnhancedExplorativeHarmonicSwarmOptimizer:
    def __init__(
        self,
        budget=10000,
        population_size=50,
        num_iterations=100,
        harmony_memory_size=10,
        bandwidth=3.0,
        exploration_rate=0.1,
    ):
        self.budget = budget
        self.population_size = population_size
        self.dim = 5
        self.num_iterations = num_iterations
        self.harmony_memory_size = harmony_memory_size
        self.bandwidth = bandwidth
        self.exploration_rate = exploration_rate

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def update_bandwidth(self, iter_count):
        return self.bandwidth / (1 + iter_count)

    def explore_new_solution(self, population, best_solution):
        exploration = np.random.uniform(
            -self.exploration_rate, self.exploration_rate, (self.population_size, self.dim)
        )
        new_population = population + exploration
        return new_population

    def __call__(self, func):
        population = self.initialize_population()
        fitness = np.array([func(sol) for sol in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        best_fitnesses = [best_fitness]

        for i in range(self.budget // self.population_size):
            new_population = self.explore_new_solution(population, best_solution)
            population = np.vstack((population, new_population))
            fitness = np.array([func(sol) for sol in population])
            sorted_indices = np.argsort(fitness)[: self.population_size]
            population = population[sorted_indices]
            fitness = np.array([func(sol) for sol in population])

            if fitness[0] < best_fitness:
                best_solution = population[0]
                best_fitness = fitness[0]

            best_fitnesses.append(best_fitness)

            self.bandwidth = self.update_bandwidth(i)

        aocc = 1 - np.std(best_fitnesses) / np.mean(best_fitnesses)
        return aocc, best_solution
