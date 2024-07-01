import numpy as np


class AdaptiveIncrementalCrossoverEnhancement:
    def __init__(self, budget, dim=5):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def initialize_population(self, num_individuals=100):
        return np.random.uniform(self.lower_bound, self.upper_bound, (num_individuals, self.dim))

    def evaluate(self, func, candidates):
        return np.array([func(ind) for ind in candidates])

    def select_survivors(self, population, fitness, top_k=20):
        indices = np.argsort(fitness)[:top_k]
        return population[indices], fitness[indices]

    def crossover(self, parents, offspring_size=50):
        num_parents = len(parents)
        offspring = np.empty((offspring_size, self.dim))
        for i in range(offspring_size):
            p1, p2 = np.random.choice(num_parents, 2, replace=False)
            cross_point = np.random.randint(1, self.dim)
            offspring[i, :cross_point] = parents[p1, :cross_point]
            offspring[i, cross_point:] = parents[p2, cross_point:]
        return offspring

    def mutate(self, population, scale=0.05):
        perturbation = np.random.normal(0, scale, size=population.shape)
        mutated = np.clip(population + perturbation, self.lower_bound, self.upper_bound)
        return mutated

    def __call__(self, func):
        population_size = 100
        elite_size = 20
        mutation_scale = 0.05
        offspring_size = 80

        population = self.initialize_population(population_size)
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

            # Selection of the elite
            elite_population, elite_fitness = self.select_survivors(population, fitness, elite_size)

            # Crossover and Mutation: generate offspring with potential greater diversity
            offspring = self.crossover(elite_population, offspring_size)
            offspring = self.mutate(offspring, mutation_scale)

            # Reinsert best found solution into population to ensure retention of good genes
            elite_population[0] = best_solution.copy()

            # Merge elite and offspring into a new population
            population = np.vstack((elite_population, offspring))

        return best_score, best_solution
