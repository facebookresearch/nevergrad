import numpy as np


class RefinedAdaptiveIncrementalCrossover:
    def __init__(self, budget, dim=5):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 150
        self.elite_size = 30
        self.offspring_size = 120
        self.mutation_scale = 0.03
        self.crossover_rate = 0.7

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))

    def evaluate(self, func, candidates):
        return np.array([func(ind) for ind in candidates])

    def select_survivors(self, population, fitness):
        indices = np.argsort(fitness)[: self.elite_size]
        return population[indices], fitness[indices]

    def crossover(self, parents):
        num_parents = len(parents)
        offspring = np.empty((self.offspring_size, self.dim))
        for i in range(self.offspring_size):
            if np.random.rand() < self.crossover_rate:
                p1, p2 = np.random.choice(num_parents, 2, replace=False)
                cross_point = np.random.randint(1, self.dim)
                offspring[i, :cross_point] = parents[p1, :cross_point]
                offspring[i, cross_point:] = parents[p2, cross_point:]
            else:
                offspring[i, :] = parents[np.random.randint(num_parents)]
        return offspring

    def mutate(self, population):
        perturbation = np.random.normal(0, self.mutation_scale, size=population.shape)
        mutated = np.clip(population + perturbation, self.lower_bound, self.upper_bound)
        return mutated

    def __call__(self, func):
        population = self.initialize_population()
        best_score = float("inf")
        best_solution = None
        evaluations_consumed = 0

        while evaluations_consumed < self.budget:
            fitness = self.evaluate(func, population)
            evaluations_consumed += len(population)

            current_best_idx = np.argmin(fitness)
            if fitness[current_best_idx] < best_score:
                best_score = fitness[current_best_idx]
                best_solution = population[current_best_idx].copy()

            if evaluations_consumed >= self.budget:
                break

            elite_population, elite_fitness = self.select_survivors(population, fitness)

            offspring = self.crossover(elite_population)
            offspring = self.mutate(offspring)

            elite_population[0] = best_solution.copy()

            population = np.vstack((elite_population, offspring))

        return best_score, best_solution
