import numpy as np


class ProgressiveDimensionalOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Initial parameter setup
        population_size = 100
        mutation_factor = 0.9  # High initial mutation rate for broad exploration
        crossover_rate = 0.7
        elite_size = 5

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(x) for x in population])
        evaluations = population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        learning_rate_decrease = self.budget / 10  # Adjust mutation and crossover rate over these intervals

        # Optimization loop
        while evaluations < self.budget:
            for i in range(population_size):
                if evaluations >= self.budget:
                    break

                # Mutation strategy adapted from DE/rand/1/bin
                indices = np.random.choice([j for j in range(population_size) if j != i], 3, replace=False)
                x1, x2, x3 = population[indices]
                mutant = x1 + mutation_factor * (x2 - x3)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                trial = np.where(np.random.rand(self.dim) < crossover_rate, mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness
                        best_index = i

            # Periodic adaptation of mutation factor and crossover rate
            if evaluations % learning_rate_decrease == 0:
                mutation_factor = max(0.5, mutation_factor * 0.95)  # Decrement mutation factor slowly
                crossover_rate = min(1.0, crossover_rate + 0.05)  # Incrementally increase crossover rate

        return best_fitness, best_solution
