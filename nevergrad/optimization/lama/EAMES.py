import numpy as np


class EAMES:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.initial_population_size = 50
        self.elite_size = 5
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7

    def initialize_population(self):
        return np.random.uniform(
            self.bounds[0], self.bounds[1], (self.initial_population_size, self.dimension)
        )

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def select_elites(self, population, fitness):
        elite_indices = np.argsort(fitness)[: self.elite_size]
        return population[elite_indices], fitness[elite_indices]

    def differential_mutation(self, population, base_idx):
        idxs = np.random.choice(self.initial_population_size, 3, replace=False)
        x1, x2, x3 = population[idxs[0]], population[idxs[1]], population[idxs[2]]
        mutant = np.clip(x1 + self.mutation_factor * (x2 - x3), self.bounds[0], self.bounds[1])
        return mutant

    def crossover(self, mutant, target):
        crossover_mask = np.random.rand(self.dimension) < self.crossover_rate
        return np.where(crossover_mask, mutant, target)

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(population, func)
        evaluations = self.initial_population_size
        iteration = 0
        best_fitness = np.inf
        best_solution = None

        while evaluations < self.budget:
            elite_population, elite_fitness = self.select_elites(population, fitness)

            for i in range(self.initial_population_size):
                mutant = self.differential_mutation(population, i)
                child = self.crossover(mutant, population[i])
                child_fitness = func(child)
                evaluations += 1

                if child_fitness < fitness[i]:
                    population[i] = child
                    fitness[i] = child_fitness

                if child_fitness < best_fitness:
                    best_fitness = child_fitness
                    best_solution = child

                if evaluations >= self.budget:
                    break

            iteration += 1

        return best_fitness, best_solution
