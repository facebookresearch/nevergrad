import numpy as np


class IASDD:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])

    def initialize(self):
        population_size = 50
        population = np.random.uniform(*self.bounds, (population_size, self.dimension))
        return population, population_size

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def adapt_search_parameters(self, std_dev):
        # Dynamically adjust search parameters based on population's standard deviation
        if std_dev < 0.1:
            return 0.1  # Narrow search
        elif std_dev < 0.5:
            return 0.3  # Moderate search
        else:
            return 0.5  # Wide search

    def __call__(self, func):
        population, population_size = self.initialize()
        best_fitness = np.Inf
        best_individual = None
        evaluations = 0

        while evaluations < self.budget:
            fitness = self.evaluate(population, func)
            evaluations += population_size

            # Update global best
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < best_fitness:
                best_fitness = fitness[min_idx]
                best_individual = population[min_idx]

            # Adapt search based on current performance
            std_dev = np.std(fitness)
            search_scale = self.adapt_search_parameters(std_dev)

            # Genetic operations: mutation and crossover
            new_population = population + np.random.normal(0, search_scale, (population_size, self.dimension))
            new_population = np.clip(new_population, *self.bounds)

            # Include best individual to ensure elitism
            new_population[np.random.randint(population_size)] = best_individual
            population = new_population

            # Dynamic diversification when stagnation detected
            if std_dev < 0.05 and evaluations < self.budget - population_size:
                diversify_count = max(5, int(population_size * 0.1))
                population[:diversify_count] = np.random.uniform(
                    *self.bounds, (diversify_count, self.dimension)
                )

        return best_fitness, best_individual
