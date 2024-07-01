import numpy as np


class RefinedAdaptiveDifferentialSpiralSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality as per problem constraints

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = np.zeros(self.dim)

        # Initialize a population around the search space
        population_size = 100  # Increased population for better exploration
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        # Initialize spiral dynamics with adaptive parameters
        radius = 5.0
        angle_increment = 2 * np.pi / population_size
        evaluations_left = self.budget - population_size
        radius_decay = 0.95  # Slower decay to maintain exploration longer
        angle_speed_increase = 1.05  # Slightly more aggressive angular increment

        while evaluations_left > 0:
            # Select indices for differential mutation using tournament selection
            tournament_size = 3
            indices = np.random.choice(population_size, tournament_size, replace=False)
            tournament_fitness = fitness[indices]
            best_index = indices[np.argmin(tournament_fitness)]
            rest_indices = np.delete(indices, np.argmin(tournament_fitness))

            # Mutation and crossover using best of tournament and random others
            a, b, c = population[best_index], population[rest_indices[0]], population[rest_indices[1]]
            mutant = a + 0.9 * (b - c)  # Increased weight for more aggressive mutations
            mutant = np.clip(mutant, -5.0, 5.0)

            # Spiral movement on the mutant
            angle = angle_increment * best_index
            offset = radius * np.array([np.cos(angle), np.sin(angle)] + [0] * (self.dim - 2))
            candidate = mutant + offset
            candidate = np.clip(candidate, -5.0, 5.0)

            # Evaluate candidate
            f_candidate = func(candidate)
            evaluations_left -= 1

            # Selection based on better fitness
            worst_index = np.argmax(fitness)
            if f_candidate < fitness[worst_index]:
                population[worst_index] = candidate
                fitness[worst_index] = f_candidate

                # Update the optimal solution found
                if f_candidate < self.f_opt:
                    self.f_opt = f_candidate
                    self.x_opt = candidate

            # Adaptive update of spiral dynamics parameters
            radius *= radius_decay
            angle_increment *= angle_speed_increase

        return self.f_opt, self.x_opt
