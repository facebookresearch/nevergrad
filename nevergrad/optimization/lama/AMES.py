import numpy as np


class AMES:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 50
        self.elite_size = 5  # Elitism parameter

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def select_elites(self, population, fitness):
        elite_indices = np.argsort(fitness)[: self.elite_size]
        return population[elite_indices], fitness[elite_indices]

    def adaptive_mutation(self, individual, iteration):
        # Decrease mutation strength as iterations increase
        mutation_strength = np.maximum(0.1, 1 - (iteration / self.budget))
        mutation_vector = np.random.normal(0, mutation_strength, self.dimension)
        return np.clip(individual + mutation_vector, self.bounds[0], self.bounds[1])

    def recombine(self, elite_population):
        # Randomly recombine pairs of elite individuals
        indices = np.random.permutation(self.elite_size)
        parent1 = elite_population[indices[0]]
        parent2 = elite_population[indices[1]]
        alpha = np.random.rand(self.dimension)
        return alpha * parent1 + (1 - alpha) * parent2

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(population, func)

        evaluations = self.population_size
        iteration = 0

        while evaluations < self.budget:
            elite_population, elite_fitness = self.select_elites(population, fitness)

            for i in range(self.population_size):
                if i < self.elite_size:
                    # Preserve elites
                    continue

                # Recombine and mutate
                child = self.recombine(elite_population)
                child = self.adaptive_mutation(child, iteration)
                child_fitness = func(child)
                evaluations += 1

                # Selection step
                if child_fitness < fitness[i]:
                    population[i] = child
                    fitness[i] = child_fitness

                if evaluations >= self.budget:
                    break

            iteration += 1

        best_idx = np.argmin(fitness)
        return fitness[best_idx], population[best_idx]
