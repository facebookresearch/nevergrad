import numpy as np


class EADE_FIDM:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population_size = 100
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

                # Enhanced Adaptive parameters based on fitness percentile
                fitness_percentile = np.argsort(np.argsort(fitness)) / population_size
                F = 0.8 * (1 - fitness_percentile[i])  # Higher mutation for better solutions
                CR = 0.9 * fitness_percentile[i]  # Higher crossover for worse solutions

                # Enhanced Mutation: Incorporating best individual
                indices = np.random.choice(np.delete(np.arange(population_size), i), 2, replace=False)
                x1, x2 = population[indices]
                mutant = population[best_idx] + F * (x1 - x2)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
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
