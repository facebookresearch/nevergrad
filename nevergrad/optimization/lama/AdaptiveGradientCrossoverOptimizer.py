import numpy as np


class AdaptiveGradientCrossoverOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality of the problem
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Optimizer parameters
        population_size = 200
        mutation_factor = 0.8  # High initial mutation factor for broader search
        crossover_rate = 0.7  # Moderately high crossover to balance exploration and exploitation
        grad_step_size = 0.01  # Step size for gradient approximation
        adaptive_rate = 0.05  # Adaptive rate for adjusting mutation and crossover

        # Initial population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        # Evolutionary process with gradient-based mutation
        while evaluations < self.budget:
            for i in range(population_size):
                # Select three different members for mutation
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Gradient-based mutation
                grad_direction = (func(a + grad_step_size) - func(a)) / grad_step_size
                mutant_vector = a + mutation_factor * grad_direction * (b - c)
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < crossover_rate
                trial_vector = np.where(crossover_mask, mutant_vector, population[i])

                trial_fitness = func(trial_vector)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial_vector
                    fitness[i] = trial_fitness

                    # Update the best solution found
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial_vector

            # Adaptive mechanism to adjust mutation and crossover rates
            mutation_factor = max(0.1, mutation_factor - adaptive_rate * np.random.randn())
            crossover_rate = min(1.0, max(0.5, crossover_rate + adaptive_rate * np.random.randn()))

        return best_fitness, best_solution
