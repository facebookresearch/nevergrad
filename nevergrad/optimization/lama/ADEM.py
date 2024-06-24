import numpy as np


class ADEM:
    def __init__(self, budget, population_size=50, crossover_rate=0.9, scaling_factor=0.8, memory_factor=0.1):
        self.budget = budget
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.scaling_factor = scaling_factor
        self.memory_factor = memory_factor
        self.dimension = 5  # Given dimensionality
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])

        # Initialize memory for best solutions
        memory = np.copy(population)
        memory_fitness = np.copy(fitness)

        # Find the best solution in the initial population
        best_index = np.argmin(fitness)
        best_fitness = fitness[best_index]
        best_individual = population[best_index]

        num_evals = self.population_size  # Initial population evaluation

        while num_evals < self.budget:
            for i in range(self.population_size):
                # Mutation using "DE/current-to-best/1" strategy
                best_idx = np.argmin(memory_fitness)
                a, b = population[np.random.choice(self.population_size, 2, replace=False)]
                mutant = (
                    population[i]
                    + self.scaling_factor * (memory[best_idx] - population[i])
                    + self.scaling_factor * (a - b)
                )
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

                    # Update memory
                    if trial_fitness < memory_fitness[i]:
                        memory[i] = trial_vector
                        memory_fitness[i] = trial_fitness

                    # Update the best found solution
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial_vector

                # Stop if budget is exhausted
                if num_evals >= self.budget:
                    break

        return best_fitness, best_individual


# Usage of ADEM:
# optimizer = ADEM(budget=1000)
# best_fitness, best_solution = optimizer(func)
