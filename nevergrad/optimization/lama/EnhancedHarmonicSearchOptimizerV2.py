import numpy as np


class EnhancedHarmonicSearchOptimizerV2:
    def __init__(
        self,
        budget=10000,
        population_size=50,
        harmony_memory_size=5,
        bandwidth=1.0,
        exploration_rate=0.1,
        memory_consideration_prob=0.7,
    ):
        self.budget = budget
        self.population_size = population_size
        self.dim = 5
        self.harmony_memory_size = harmony_memory_size
        self.bandwidth = bandwidth
        self.exploration_rate = exploration_rate
        self.memory_consideration_prob = memory_consideration_prob

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def update_bandwidth(self, iter_count):
        return self.bandwidth / (1 + iter_count)

    def explore_new_solution(self, population, best_solution, bandwidth):
        exploration = np.random.normal(0, bandwidth, size=(self.population_size, self.dim))
        new_population = population + exploration
        return new_population

    def update_harmony_memory(self, harmony_memory, new_solution, fitness):
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < harmony_memory[-1][1]:
            harmony_memory[-1] = (new_solution[min_idx], fitness[min_idx])
        return harmony_memory

    def adaptive_bandwidth(self, best_fitness, prev_best_fitness, bandwidth):
        if best_fitness < prev_best_fitness:
            return bandwidth * 1.05
        else:
            return bandwidth * 0.95

    def __call__(self, func):
        population = self.initialize_population()
        fitness = np.array([func(sol) for sol in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        harmony_memory = [(best_solution, best_fitness)]

        best_fitnesses = [best_fitness]
        prev_best_fitness = best_fitness

        for i in range(self.budget // self.population_size):
            new_population = self.explore_new_solution(population, best_solution, self.bandwidth)
            population = new_population
            fitness = np.array([func(sol) for sol in population])

            if np.random.rand() < self.memory_consideration_prob:
                harmony_memory = self.update_harmony_memory(harmony_memory, population, fitness)
                population = np.vstack([h[0] for h in harmony_memory])

            if fitness[0] < best_fitness:
                best_solution = population[0]
                best_fitness = fitness[0]

            best_fitnesses.append(best_fitness)

            self.bandwidth = self.adaptive_bandwidth(best_fitness, prev_best_fitness, self.bandwidth)
            prev_best_fitness = best_fitness

        aocc = 1 - np.std(best_fitnesses) / np.mean(best_fitnesses)
        return aocc, best_solution
