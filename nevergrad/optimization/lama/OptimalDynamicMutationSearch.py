import numpy as np


class OptimalDynamicMutationSearch:
    def __init__(
        self,
        budget,
        population_size=50,
        crossover_rate=0.9,
        F_min=0.5,
        F_max=1.2,
        memory_size=30,
        elite_size=5,
    ):
        self.budget = budget
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.F_min = F_min
        self.F_max = F_max
        self.memory_size = memory_size
        self.elite_size = elite_size

    def __call__(self, func):
        lb, ub, dimension = -5.0, 5.0, 5

        # Initialize population and fitness
        population = lb + (ub - lb) * np.random.rand(self.population_size, dimension)
        fitness = np.array([func(individual) for individual in population])

        # Initialize memory
        memory = np.empty((self.memory_size, dimension))
        memory_fitness = np.full(self.memory_size, np.inf)

        # Track the best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            # Update memory
            for i in range(self.population_size):
                if fitness[i] < np.max(memory_fitness):
                    worst_idx = np.argmax(memory_fitness)
                    memory[worst_idx] = population[i]
                    memory_fitness[worst_idx] = fitness[i]

            # Mutation and Crossover
            for i in range(self.population_size):
                # Adaptive mutation factor based on population diversity
                F = self.F_min + (self.F_max - self.F_min) * (1 - np.std(fitness) / np.ptp(fitness))

                # Select indices for mutation
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                # Mutation: DE/rand-to-best/1
                mutant = np.clip(population[i] + F * (best_solution - population[i] + a - b), lb, ub)

                # Crossover
                cross_points = np.random.rand(dimension) < self.crossover_rate
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update the best solution found
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
