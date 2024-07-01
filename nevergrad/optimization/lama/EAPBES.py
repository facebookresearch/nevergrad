import numpy as np


class EAPBES:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 50
        self.elite_size = 5  # Top 10% as elite
        self.mutation_rate = 0.1
        self.crossover_probability = 0.5

    def initialize(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def select_elites(self, population, fitness):
        elite_indices = np.argsort(fitness)[: self.elite_size]
        return population[elite_indices], fitness[elite_indices]

    def mutate(self, individual):
        mutation_mask = np.random.rand(self.dimension) < self.mutation_rate
        individual[mutation_mask] += np.random.normal(0, 0.1, np.sum(mutation_mask))
        return np.clip(individual, self.bounds[0], self.bounds[1])

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_probability:
            point = np.random.randint(self.dimension)
            child = np.concatenate([parent1[:point], parent2[point:]])
        else:
            mask = np.random.rand(self.dimension) < 0.5
            child = parent1 * mask + parent2 * (1 - mask)
        return child

    def local_search(self, elite):
        perturbation = np.random.normal(0, 0.05, self.dimension)  # smaller scale perturbation
        candidate = elite + perturbation
        return np.clip(candidate, self.bounds[0], self.bounds[1])

    def __call__(self, func):
        population = self.initialize()
        best_fitness = np.inf
        best_individual = None

        evaluations = 0
        generations = 0
        while evaluations < self.budget:
            fitness = self.evaluate(population, func)
            evaluations += len(population)

            if np.min(fitness) < best_fitness:
                best_fitness = np.min(fitness)
                best_individual = population[np.argmin(fitness)].copy()

            elites, elite_fitness = self.select_elites(population, fitness)

            if generations % 10 == 0 and generations > 0:  # Local search every 10 generations
                for i in range(len(elites)):
                    elites[i] = self.local_search(elites[i])

            new_population = elites.copy()
            while len(new_population) < self.population_size:
                parent1, parent2 = population[np.random.choice(len(population), 2, replace=False)]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population = np.vstack([new_population, child])

            population = new_population
            generations += 1

        return best_fitness, best_individual
