import numpy as np


class AdaptiveHarmonyFireworksAlgorithm:
    def __init__(
        self,
        budget=10000,
        population_size=30,
        harmony_memory_size=10,
        pitch_adjust_rate=0.5,
        mutation_rate=0.1,
    ):
        self.budget = budget
        self.population_size = population_size
        self.dim = 5
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjust_rate = pitch_adjust_rate
        self.mutation_rate = mutation_rate

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def pitch_adjustment(self, solution, best_solution):
        new_solution = solution.copy()
        for i in range(self.dim):
            if np.random.rand() < self.pitch_adjust_rate:
                new_solution[i] = best_solution[i]

        return new_solution

    def fireworks_mutation(self, solution):
        new_solution = solution + self.mutation_rate * np.random.normal(0, 1, self.dim)

        return np.clip(new_solution, -5.0, 5.0)

    def __call__(self, func):
        population = self.initialize_population()
        memory = population[
            np.random.choice(range(self.population_size), self.harmony_memory_size, replace=False)
        ]
        fitness = [func(sol) for sol in population]
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        for _ in range(self.budget // self.population_size):
            new_solution = self.pitch_adjustment(
                population[np.random.randint(self.population_size)], best_solution
            )
            new_solution = self.fireworks_mutation(new_solution)
            population = np.vstack((population, new_solution))
            fitness = [func(sol) for sol in population]
            sorted_indices = np.argsort(fitness)[: self.population_size]
            population = population[sorted_indices]
            fitness = [func(sol) for sol in population]

            if fitness[0] < best_fitness:
                best_solution = population[0]
                best_fitness = fitness[0]

            memory = np.vstack((memory, population[: self.harmony_memory_size]))
            memory_fitness = [func(sol) for sol in memory]
            memory_sorted_indices = np.argsort(memory_fitness)[: self.harmony_memory_size]
            memory = memory[memory_sorted_indices]

        return best_fitness, best_solution
