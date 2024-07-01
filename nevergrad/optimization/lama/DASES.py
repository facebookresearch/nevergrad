import numpy as np


class DASES:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 50
        self.elite_size = 5  # Top 10% as elite

    def initialize(self):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))
        mutation_rates = np.random.rand(self.population_size) * 0.1
        crossover_rates = np.random.rand(self.population_size) * 0.8 + 0.2
        return population, mutation_rates, crossover_rates

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutate(self, individual, mutation_rate):
        mutation_mask = np.random.rand(self.dimension) < mutation_rate
        individual[mutation_mask] += np.random.normal(0, 1, np.sum(mutation_mask))
        return np.clip(individual, self.bounds[0], self.bounds[1])

    def crossover(self, parent1, parent2, crossover_rate):
        child = np.where(np.random.rand(self.dimension) < crossover_rate, parent1, parent2)
        return child

    def select_elites(self, population, fitness):
        elite_indices = np.argsort(fitness)[: self.elite_size]
        return population[elite_indices], fitness[elite_indices]

    def local_search(self, elite):
        perturbation = np.random.normal(0, 0.05, self.dimension)
        candidate = elite + perturbation
        return np.clip(candidate, self.bounds[0], self.bounds[1])

    def __call__(self, func):
        population, mutation_rates, crossover_rates = self.initialize()
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
            new_population = elites.tolist()  # Start next gen with elites

            while len(new_population) < self.population_size:
                idx1, idx2 = np.random.choice(self.population_size, 2, replace=False)
                child = self.crossover(population[idx1], population[idx2], crossover_rates[idx1])
                child = self.mutate(child, mutation_rates[idx1])
                new_population.append(child)

            population = np.array(new_population)
            if evaluations // self.population_size % 5 == 0:
                for i in range(len(elites)):
                    elites[i] = self.local_search(elites[i])

        return best_fitness, best_individual
