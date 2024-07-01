import numpy as np


class RefinedAdaptiveSpatialExplorationOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population_size = 150  # Increased population size for better exploration
        mutation_factor = 0.8  # More aggressive initial mutation
        crossover_rate = 0.7  # Initial crossover rate
        elite_size = 10  # Increased elite size

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(x) for x in population])
        evaluations = population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        # Main loop
        while evaluations < self.budget:
            new_population = np.empty_like(population)
            new_fitness = np.empty_like(fitness)

            # Elitism: carry forward best solutions
            sorted_indices = np.argsort(fitness)[:elite_size]
            new_population[:elite_size] = population[sorted_indices]
            new_fitness[:elite_size] = fitness[sorted_indices]

            # Generate new candidates for the rest of the population
            for i in range(elite_size, population_size):
                # Tournament selection
                idxs = np.random.choice(population_size, 3, replace=False)
                if fitness[idxs[0]] < fitness[idxs[1]]:
                    better_idx = idxs[0]
                else:
                    better_idx = idxs[1]

                target = population[better_idx]
                a, b, c = population[np.random.choice(population_size, 3, replace=False)]
                mutant = a + mutation_factor * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Binomial crossover
                cross_points = np.random.rand(self.dim) < crossover_rate
                trial = np.where(cross_points, mutant, target)
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                new_population[i] = trial if trial_fitness < fitness[better_idx] else population[better_idx]
                new_fitness[i] = trial_fitness if trial_fitness < fitness[better_idx] else fitness[better_idx]

            population = new_population
            fitness = new_fitness

            # Update best solution if found
            current_best_index = np.argmin(fitness)
            if fitness[current_best_index] < best_fitness:
                best_fitness = fitness[current_best_index]
                best_solution = population[current_best_index]

            # Update parameters dynamically
            if evaluations % (self.budget // 10) == 0:
                mutation_factor *= 0.95  # Gradually reduce mutation to stabilize convergence
                crossover_rate *= 0.98  # Slowly reduce the crossover rate

        return best_fitness, best_solution
