import numpy as np


class EnhancedAdaptiveDualStrategyOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality of the problem
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Optimizer parameters
        population_size = 100
        mutation_factor = 0.5  # Constant mutation factor
        crossover_rate = 0.9  # Increased crossover for better exploration
        elite_size = 5  # Same number of elite individuals to preserve

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        # Main optimization loop
        while evaluations < self.budget:
            new_population = np.zeros_like(population)
            new_fitness = np.zeros_like(fitness)

            # Preserve elite solutions
            elite_indices = np.argsort(fitness)[:elite_size]
            new_population[:elite_size] = population[elite_indices]
            new_fitness[:elite_size] = fitness[elite_indices]

            # Generate the rest of the new population
            for i in range(elite_size, population_size):
                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = population[indices]

                # Adaptive mutation based on relative fitness differences
                adaptive_mutation = mutation_factor * (1 + (fitness[i] - best_fitness) / best_fitness)

                mutant = a + adaptive_mutation * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                trial_vector = np.where(np.random.rand(self.dim) < crossover_rate, mutant, population[i])
                trial_fitness = func(trial_vector)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    new_population[i] = trial_vector
                    new_fitness[i] = trial_fitness
                else:
                    new_population[i] = population[i]
                    new_fitness[i] = fitness[i]

                # Update best solution found
                if new_fitness[i] < best_fitness:
                    best_fitness = new_fitness[i]
                    best_solution = new_population[i]

            population = new_population
            fitness = new_fitness

        return best_fitness, best_solution
