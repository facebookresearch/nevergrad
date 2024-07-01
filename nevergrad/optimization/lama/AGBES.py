import numpy as np


class AGBES:
    def __init__(self, budget, population_size=100, gradient_weight=0.3, mutation_rate=0.1, elite_ratio=0.2):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0
        self.gradient_weight = gradient_weight
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        num_evals = self.population_size

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()

        # Main evolutionary loop
        while num_evals < self.budget:
            elite_count = int(self.population_size * self.elite_ratio)
            elite_indices = np.argsort(fitness)[:elite_count]
            elite_individuals = population[elite_indices]

            # Generate new population
            new_population = np.zeros_like(population)

            # Reproduction with mutation and gradient information
            for i in range(self.population_size):
                if i < elite_count:
                    # Elites undergo mutation only
                    mutation = np.random.randn(self.dimension) * self.mutation_rate
                    new_individual = elite_individuals[i % elite_count] + mutation
                else:
                    # Non-elites are generated from random elite and gradient information
                    parent = elite_individuals[np.random.randint(0, elite_count)]
                    gradient = best_individual - parent
                    perturbation = np.random.randn(self.dimension) * self.mutation_rate
                    new_individual = parent + self.gradient_weight * gradient + perturbation

                new_individual = np.clip(new_individual, self.lb, self.ub)
                new_population[i] = new_individual

            # Evaluate new population
            new_fitness = np.array([func(ind) for ind in new_population])
            num_evals += self.population_size

            # Selection process: Elitism combined with direct competition
            for j in range(self.population_size):
                if new_fitness[j] < fitness[j]:
                    population[j] = new_population[j]
                    fitness[j] = new_fitness[j]
                    if new_fitness[j] < best_fitness:
                        best_fitness = new_fitness[j]
                        best_individual = new_population[j].copy()

        return best_fitness, best_individual
