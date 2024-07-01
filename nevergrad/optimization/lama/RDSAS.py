import numpy as np


class RDSAS:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])

    def initialize(self):
        population_size = 50
        population = np.random.uniform(self.bounds[0], self.bounds[1], (population_size, self.dimension))
        return population, population_size

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def local_search(self, individual, func, base_fitness):
        # Safe scale computation with minimum boundary
        scale = max(abs(base_fitness) * 0.01, 0.1)  # Ensuring scale is never zero or negative
        perturbation = np.random.normal(0, scale, self.dimension)
        new_individual = individual + perturbation
        new_individual = np.clip(new_individual, self.bounds[0], self.bounds[1])
        new_fitness = func(new_individual)
        return new_individual if new_fitness < base_fitness else individual

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
                best_individual = population[min_idx].copy()

            # Local search with dynamic scale adjustment
            for i in range(population_size):
                if np.random.rand() < 0.1:  # 10% chance to perform local search
                    population[i] = self.local_search(population[i], func, fitness[i])

            # Enhanced synchronization based on performance stagnation
            if evaluations % (100 + int(1000 * (best_fitness / (np.mean(fitness) + 1e-6)))) == 0:
                population[np.random.randint(population_size)] = best_individual.copy()

            # Continuous diversification
            if np.random.rand() < 0.02:  # 2% chance for diversification
                idx_to_diversify = np.random.randint(population_size)
                population[idx_to_diversify] = np.random.uniform(
                    self.bounds[0], self.bounds[1], self.dimension
                )

        return best_fitness, best_individual
