import numpy as np


class AdaptivePrecisionRotationalClimbOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimension fixed as per problem statement
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population_size = 30  # Reduced population for more focused exploration
        mutation_rate = 0.08  # Further reduced mutation rate for finer adjustments
        rotation_rate = 0.03  # Adaptive rotation rate for small precise rotations
        blend_factor = 0.7  # Increased blend factor for stronger pull towards better solutions

        # Initialize population within the bounds
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(x) for x in population])
        evaluations = population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        while evaluations < self.budget:
            for i in range(population_size):
                # Select mutation indices ensuring unique entries
                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = population[indices]

                # Mutation and rotational operation
                direction = b - c
                theta = rotation_rate * 2 * np.pi  # Complete rotation consideration
                rotation_matrix = np.eye(self.dim)
                if self.dim >= 2:  # Ensure rotation is only applied if dimensionality permits
                    np.fill_diagonal(rotation_matrix[:2, :2], np.cos(theta))
                    rotation_matrix[0, 1], rotation_matrix[1, 0] = -np.sin(theta), np.sin(theta)

                rotated_vector = np.dot(rotation_matrix, direction)
                mutant = a + mutation_rate * rotated_vector
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover with adaptive precision
                trial = best_solution + blend_factor * (mutant - best_solution)
                trial = np.clip(trial, self.lower_bound, self.upper_bound)
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial

            # Adaptive adjustment of rotation and mutation parameters based on progress
            if evaluations % (self.budget // 10) == 0:  # Every 10% of the budget
                rotation_rate *= 0.95  # Gradually decrease rotation rate
                mutation_rate *= 0.95  # Gradually decrease mutation rate
                blend_factor = min(blend_factor + 0.02, 1.0)  # Gradually increase blend factor to 1.0

        return best_fitness, best_solution
