import numpy as np


class AdaptiveRotationalClimbOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the search space
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population_size = 150  # Adjusted population size for balance
        mutation_rate = 0.1  # Base mutation rate
        rotation_rate = 0.05  # Rotation applied to the difference vectors
        alpha = 0.9  # Factor for blending the mutant vector with the current best

        # Initialize population and evaluate
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(x) for x in population])
        evaluations = population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        while evaluations < self.budget:
            for i in range(population_size):
                # Select random indices for mutation
                idxs = np.random.choice(population_size, 3, replace=False)
                a, b, c = population[idxs]

                # Perform mutation with rotational component
                direction = b - c
                theta = rotation_rate * np.pi
                rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
                if self.dim > 2:
                    extended_rotation = np.eye(self.dim)
                    extended_rotation[:2, :2] = rotation_matrix
                else:
                    extended_rotation = rotation_matrix

                rotated_direction = np.dot(extended_rotation, direction)
                mutant = a + mutation_rate * rotated_direction
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover current best with mutant
                trial = best_solution + alpha * (mutant - best_solution)
                trial = np.clip(trial, self.lower_bound, self.upper_bound)
                trial_fitness = func(trial)
                evaluations += 1

                # Selection
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update best solution found
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial

        return best_fitness, best_solution
