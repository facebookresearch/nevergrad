import numpy as np


class DifferentialGradientEvolutionStrategy:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.lb = -5.0  # Lower bound for each dimension
        self.ub = 5.0  # Upper bound for each dimension

    def __call__(self, func):
        # Initialization parameters
        population_size = 50
        mutation_factor = 0.8
        crossover_prob = 0.7
        # Initialize the population
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])

        # Track the best solution
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx].copy()

        evaluations = population_size

        while evaluations < self.budget:
            for i in range(population_size):
                # Differential evolution mutation and crossover
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + mutation_factor * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < crossover_prob
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                # Update the best found solution
                if trial_fitness < self.f_opt:
                    self.f_opt = trial_fitness
                    self.x_opt = trial.copy()

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt


# Example of usage:
# optimizer = DifferentialGradientEvolutionStrategy(budget=10000)
# best_value, best_solution = optimizer(func)
