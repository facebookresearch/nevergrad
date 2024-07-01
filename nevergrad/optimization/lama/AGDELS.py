import numpy as np


class AGDELS:
    def __init__(self, budget, population_size=100, F_base=0.8, CR_base=0.9, local_search_prob=0.1):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lb = np.full(self.dimension, -5.0)
        self.ub = np.full(self.dimension, 5.0)
        self.F_base = F_base
        self.CR_base = CR_base
        self.local_search_prob = local_search_prob

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        fitness = np.array([func(individual) for individual in population])
        num_evals = self.population_size

        # Track the best solution
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()

        while num_evals < self.budget:
            new_population = np.empty_like(population)

            for i in range(self.population_size):
                if num_evals >= self.budget:
                    break

                # Adaptive mutation parameters with Gaussian perturbation
                F = np.clip(np.random.normal(self.F_base, 0.1), 0.1, 1.0)
                CR = np.clip(np.random.normal(self.CR_base, 0.05), 0.1, 1.0)

                # Mutation and crossover
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = a + F * (b - c)
                mutant = np.clip(mutant, self.lb, self.ub)
                cross_points = np.random.rand(self.dimension) < CR
                trial = np.where(cross_points, mutant, population[i])

                # Performing local search with a certain probability
                if np.random.rand() < self.local_search_prob:
                    local_point = trial + np.random.normal(0, 0.1, self.dimension)
                    local_point = np.clip(local_point, self.lb, self.ub)
                    local_fitness = func(local_point)
                    num_evals += 1
                    if local_fitness < fitness[i]:
                        trial = local_point
                        trial_fitness = local_fitness
                    else:
                        trial_fitness = func(trial)
                        num_evals += 1
                else:
                    trial_fitness = func(trial)
                    num_evals += 1

                # Selection
                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()
                else:
                    new_population[i] = population[i]

            population = new_population

        return best_fitness, best_individual
