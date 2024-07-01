import numpy as np


class OptimizedDynamicAdaptiveHybridOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Configuration
        population_size = 300
        initial_mutation_factor = 0.5  # Moderately aggressive initial mutation
        initial_crossover_prob = 0.9  # Initially very high crossover probability
        adaptive_factor_mut = 0.001  # Finer adaptive change for mutation factor
        adaptive_factor_cross = 0.001  # Finer adaptive change for crossover probability

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_value = fitness[best_idx]

        num_iterations = self.budget // population_size
        mutation_factor = initial_mutation_factor
        crossover_prob = initial_crossover_prob

        for iteration in range(num_iterations):
            for i in range(population_size):
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                # Dynamically adjust mutation and crossover based on performance every 5 iterations
                if iteration % 5 == 0:
                    current_mean_fitness = np.mean(fitness)
                    if best_value < current_mean_fitness:
                        mutation_factor = max(0.1, mutation_factor + adaptive_factor_mut)
                        crossover_prob = max(0.1, crossover_prob - adaptive_factor_cross)
                    else:
                        mutation_factor = max(0.1, mutation_factor - adaptive_factor_mut)
                        crossover_prob = min(1.0, crossover_prob + adaptive_factor_cross)

                mutant = np.clip(a + mutation_factor * (b - c), self.lower_bound, self.upper_bound)
                trial = np.where(np.random.rand(self.dim) < crossover_prob, mutant, population[i])
                trial_fitness = func(trial)

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_value:
                        best_value = trial_fitness
                        best_solution = trial

        return best_value, best_solution
