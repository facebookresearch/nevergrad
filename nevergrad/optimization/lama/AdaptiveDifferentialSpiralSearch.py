import numpy as np


class AdaptiveDifferentialSpiralSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality as per problem constraints

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = np.zeros(self.dim)

        # Initialize a population around the search space
        population_size = 50
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        # Initialize spiral dynamics
        radius = 5.0
        angle_increment = 2 * np.pi / population_size
        evaluations_left = self.budget - population_size
        radius_decay = 0.98
        angle_speed_increase = 1.03

        while evaluations_left > 0:
            # Select three random indices for differential mutation
            indices = np.random.choice(population_size, 3, replace=False)
            a, b, c = population[indices]

            # Mutation and crossover
            mutant = a + 0.8 * (b - c)
            mutant = np.clip(mutant, -5.0, 5.0)

            # Spiral step modification on the mutant
            angle = np.random.uniform(0, 2 * np.pi)
            offset = radius * np.array([np.cos(angle), np.sin(angle)] + [0] * (self.dim - 2))
            candidate = mutant + offset
            candidate = np.clip(candidate, -5.0, 5.0)

            # Evaluate candidate
            f_candidate = func(candidate)
            evaluations_left -= 1

            # Selection
            worst_index = np.argmax(fitness)
            if f_candidate < fitness[worst_index]:
                population[worst_index] = candidate
                fitness[worst_index] = f_candidate

                # Update the optimal solution found
                if f_candidate < self.f_opt:
                    self.f_opt = f_candidate
                    self.x_opt = candidate

            # Update spiral dynamics parameters
            radius *= radius_decay
            angle_increment *= angle_speed_increase

        return self.f_opt, self.x_opt
