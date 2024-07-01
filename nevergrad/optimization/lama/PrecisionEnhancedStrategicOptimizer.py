import numpy as np


class PrecisionEnhancedStrategicOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality of the problem
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Optimizer parameters
        population_size = 100  # Slightly reduced population for increased iterations within budget
        mutation_factor = 0.9  # Refined mutation factor for a balance between exploration and exploitation
        crossover_rate = 0.85  # Increased crossover rate to foster more diverse gene combinations
        elite_size = 5  # Reduced elite size to promote diversity

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

                # Mutation
                mutant = a + mutation_factor * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Binomial crossover
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
