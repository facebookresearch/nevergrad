import numpy as np


class ADE_FPC:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population_size = 50
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        num_evals = population_size

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()

        while num_evals < self.budget:
            for i in range(population_size):
                if num_evals >= self.budget:
                    break

                # Adaptive parameters based on normalized fitness values
                norm_fitness = (fitness - np.min(fitness)) / (np.max(fitness) - np.min(fitness) + 1e-10)
                F = 0.5 + 0.5 * norm_fitness[i]  # Higher mutation for worse solutions
                CR = 0.5 * (1 - norm_fitness[i])  # Higher crossover for better solutions

                # Mutation and Crossover
                indices = np.random.choice(np.delete(np.arange(population_size), i), 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = x1 + F * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                cross_points = np.random.rand(self.dimension) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dimension)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                num_evals += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()

        return best_fitness, best_individual
