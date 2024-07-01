import numpy as np


class EPDE:
    def __init__(self, budget, population_size=50, f_min=0.5, f_max=0.8, cr_min=0.2, cr_max=0.5):
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
                # Adjust control parameters based on a non-linear scale (quadratic adjustments)
                remaining_budget = self.budget - num_evals
                cr = self.cr_min + (self.cr_max - self.cr_min) * ((remaining_budget / self.budget) ** 2)
                f = self.f_min + (self.f_max - self.f_min) * ((remaining_budget / self.budget) ** 2)

                # Mutation: DE/rand-to-best/2/bin strategy
                indices = np.random.choice(self.population_size, 4, replace=False)
                x1, x2, x3, x4 = population[indices]
                mutant = np.clip(best_individual + f * (x2 - x3 + x4 - x1), self.lb, self.ub)

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


# Usage of EPDE:
# optimizer = EPDE(budget=1000)
# best_fitness, best_solution = optimizer(func)
