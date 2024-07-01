import numpy as np


class AdaptiveSpatialExplorationOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimension fixed as per problem statement
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population_size = 100  # Increased population for broader coverage
        mutation_factor = 0.5  # Initial higher mutation factor for aggressive exploration
        crossover_rate = 0.9  # High crossover to encourage information sharing
        elite_size = 5  # Number of top solutions to keep unchanged

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(x) for x in population])
        evaluations = population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        while evaluations < self.budget:
            new_population = np.empty_like(population)
            new_fitness = np.empty_like(fitness)

            # Sort by fitness and adopt elitism
            sorted_indices = np.argsort(fitness)
            for i in range(elite_size):
                idx = sorted_indices[i]
                new_population[i] = population[idx]
                new_fitness[i] = fitness[idx]

            # Generate the rest of the new population
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

                # Crossover
                cross_points = np.random.rand(self.dim) < crossover_rate
                trial = np.where(cross_points, mutant, target)
                trial_fitness = func(trial)
                evaluations += 1

                new_population[i] = trial if trial_fitness < fitness[better_idx] else population[better_idx]
                new_fitness[i] = trial_fitness if trial_fitness < fitness[better_idx] else fitness[better_idx]

            population = new_population
            fitness = new_fitness

            current_best_index = np.argmin(fitness)
            if fitness[current_best_index] < best_fitness:
                best_fitness = fitness[current_best_index]
                best_solution = population[current_best_index]

            # Dynamically adjust mutation factor and crossover rate based on progress
            if evaluations % (self.budget // 10) == 0:
                mutation_factor *= 0.9
                crossover_rate *= 0.95

        return best_fitness, best_solution
