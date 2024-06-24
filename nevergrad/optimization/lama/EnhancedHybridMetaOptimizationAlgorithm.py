import numpy as np


class EnhancedHybridMetaOptimizationAlgorithm:
    def __init__(
        self,
        budget=10000,
        population_size=50,
        harmony_memory_size=20,
        pitch_adjust_rate=0.7,
        mutation_rate=0.2,
        diversity_rate=0.3,
        num_cuckoos=10,
        step_size=0.1,
    ):
        self.budget = budget
        self.population_size = population_size
        self.dim = 5
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjust_rate = pitch_adjust_rate
        self.mutation_rate = mutation_rate
        self.diversity_rate = diversity_rate
        self.num_cuckoos = num_cuckoos
        self.step_size = step_size

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def pitch_adjustment(self, solution, best_solution):
        new_solution = solution.copy()
        for i in range(self.dim):
            if np.random.rand() < self.pitch_adjust_rate:
                if np.random.rand() < 0.5:
                    new_solution[i] = best_solution[i]
                else:
                    new_solution[i] = np.random.uniform(-5.0, 5.0)

        return new_solution

    def fireworks_mutation(self, solution):
        new_solution = solution + self.mutation_rate * np.random.normal(0, 1, self.dim)
        return np.clip(new_solution, -5.0, 5.0)

    def cuckoo_search(self, solution):
        cuckoo = solution + self.step_size * np.random.normal(0, 1, self.dim)
        return np.clip(cuckoo, -5.0, 5.0)

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

            for _ in range(self.num_cuckoos):
                cuckoo_solution = self.cuckoo_search(population[np.random.randint(len(population))])
                population = np.vstack((population, cuckoo_solution))

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

            if np.random.rand() < self.diversity_rate:
                population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

        return best_fitness, best_solution
