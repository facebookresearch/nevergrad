import numpy as np


class PDEAF:
    def __init__(self, budget, population_size=50, f_min=0.1, f_max=0.9, cr_min=0.1, cr_max=0.9):
        self.budget = budget
        self.population_size = population_size
        self.f_min = f_min  # Minimum scaling factor
        self.f_max = f_max  # Maximum scaling factor
        self.cr_min = cr_min  # Minimum crossover probability
        self.cr_max = cr_max  # Maximum crossover probability
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

        while num_evals < self.budget:
            for i in range(self.population_size):
                # Adapt control parameters linearly based on remaining budget
                remaining_budget = self.budget - num_evals
                cr = self.cr_min + (self.cr_max - self.cr_min) * (remaining_budget / self.budget)
                f = self.f_min + (self.f_max - self.f_min) * (remaining_budget / self.budget)

                # Mutation: DE/rand/1/bin strategy
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = np.clip(x1 + f * (x2 - x3), self.lb, self.ub)

                # Crossover
                crossover_mask = np.random.rand(self.dimension) < cr
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


# Usage of PDEAF:
# optimizer = PDEAF(budget=1000)
# best_fitness, best_solution = optimizer(func)
