import numpy as np


class IncrementalCrossoverOptimization:
    def __init__(self, budget, dim=5):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def initialize_population(self, num_individuals=50):
        return np.random.uniform(self.lower_bound, self.upper_bound, (num_individuals, self.dim))

    def evaluate(self, func, candidates):
        return np.array([func(ind) for ind in candidates])

    def select_survivors(self, population, fitness, top_k=10):
        indices = np.argsort(fitness)[:top_k]
        return population[indices], fitness[indices]

    def crossover(self, parents):
        num_parents = len(parents)
        offspring = np.empty_like(parents)
        for i in range(len(parents)):
            p1, p2 = np.random.choice(num_parents, 2, replace=False)
            cross_point = np.random.randint(1, self.dim)
            offspring[i, :cross_point] = parents[p1, :cross_point]
            offspring[i, cross_point:] = parents[p2, cross_point:]
        return offspring

    def mutate(self, population, scale=0.1):
        perturbation = np.random.normal(0, scale, size=population.shape)
        mutated = np.clip(population + perturbation, self.lower_bound, self.upper_bound)
        return mutated

    def __call__(self, func):
        population_size = 50
        elite_size = 10
        mutation_scale = 0.1

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

            # Selection
            elite_population, elite_fitness = self.select_survivors(population, fitness, elite_size)

            # Crossover and Mutation
            offspring = self.crossover(elite_population)
            offspring = self.mutate(offspring, mutation_scale)

            # Reinsert best found solution into population to maintain good genes
            offspring[-1] = best_solution.copy()

            # Update population
            population = offspring

        return best_score, best_solution
