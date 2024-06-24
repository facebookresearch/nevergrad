import numpy as np


class APBES:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 50
        self.elite_size = 5  # Top 10% as elite

    def initialize(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def select_elites(self, population, fitness):
        elite_indices = np.argsort(fitness)[: self.elite_size]
        return population[elite_indices], fitness[elite_indices]

    def mutate(self, individual):
        mutation_rate = 0.1  # 10% mutation rate
        mutation_mask = np.random.rand(self.dimension) < mutation_rate
        individual[mutation_mask] += np.random.normal(0, 0.1, np.sum(mutation_mask))
        return np.clip(individual, self.bounds[0], self.bounds[1])

    def crossover(self, parent1, parent2):
        if np.random.rand() < 0.5:  # 50% chance for one-point crossover
            point = np.random.randint(self.dimension)
            child = np.concatenate([parent1[:point], parent2[point:]])
        else:  # uniform crossover
            mask = np.random.rand(self.dimension) < 0.5
            child = parent1 * mask + parent2 * (1 - mask)
        return child

    def __call__(self, func):
        population = self.initialize()
        best_fitness = np.inf
        best_individual = None

        evaluations = 0
        while evaluations < self.budget:
            fitness = self.evaluate(population, func)
            evaluations += len(population)

            if np.min(fitness) < best_fitness:
                best_fitness = np.min(fitness)
                best_individual = population[np.argmin(fitness)].copy()

            elites, elite_fitness = self.select_elites(population, fitness)

            # Generate new population
            new_population = elites.copy()
            while len(new_population) < self.population_size:
                parent1, parent2 = population[np.random.choice(len(population), 2, replace=False)]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population = np.vstack([new_population, child])

            population = new_population

        return best_fitness, best_individual
