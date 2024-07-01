import numpy as np


class DSDE:
    def __init__(self, budget, population_size=50, crossover_rate=0.8, base_scaling_factor=0.5):
        self.budget = budget
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.base_scaling_factor = base_scaling_factor
        self.dimension = 5  # Given dimensionality
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])

        # Find the best solution in the initial population
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx]

        num_evals = self.population_size  # Initial population evaluation

        while num_evals < self.budget:
            for i in range(self.population_size):
                # Mutate using a dynamic scaling factor that adapts with progress
                progress = num_evals / self.budget
                scaling_factor = self.base_scaling_factor + (1 - progress) * (0.9 - self.base_scaling_factor)

                # Mutation using the "DE/rand/1" strategy
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = x1 + scaling_factor * (x2 - x3)
                mutant = np.clip(mutant, self.lb, self.ub)  # Ensure mutant is within bounds

                # Crossover
                cross_points = np.random.rand(self.dimension) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dimension)] = True
                trial_vector = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial_vector)
                num_evals += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

                    # Update the best found solution
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial_vector

                # Stop if budget is exhausted
                if num_evals >= self.budget:
                    break

        return best_fitness, best_individual


# Usage of DSDE:
# optimizer = DSDE(budget=1000)
# best_fitness, best_solution = optimizer(func)
