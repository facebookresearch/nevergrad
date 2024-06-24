import numpy as np


class DMES:
    def __init__(self, budget, population_size=50, f_initial=0.5, cr_initial=0.5):
        self.budget = budget
        self.population_size = population_size
        self.f_initial = f_initial  # Initial scaling factor
        self.cr_initial = cr_initial  # Initial crossover rate
        self.dimension = 5
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx]

        num_evals = self.population_size

        # Initialize mutation and crossover factors dynamically
        f = np.full(self.population_size, self.f_initial)
        cr = np.full(self.population_size, self.cr_initial)

        while num_evals < self.budget:
            for i in range(self.population_size):
                # Dynamic adjustment of f and cr based on individual performance
                f[i] = np.clip(self.f_initial * (1 - (fitness[i] - best_fitness)), 0.1, 0.9)
                cr[i] = np.clip(self.cr_initial * (1 - (fitness[i] - best_fitness)), 0.1, 0.9)

                indices = np.random.choice(self.population_size, 3, replace=False)
                x0, x1, x2 = population[indices]

                # Mutation
                mutant = x0 + f[i] * (x1 - x2)
                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover
                crossover_mask = np.random.rand(self.dimension) < cr[i]
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dimension)] = True
                trial_vector = np.where(crossover_mask, mutant, population[i])

                # Selection
                trial_fitness = func(trial_vector)
                num_evals += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial_vector

                if num_evals >= self.budget:
                    break

        return best_fitness, best_individual


# Usage of DMES:
# optimizer = DMES(budget=1000)
# best_fitness, best_solution = optimizer(func)
