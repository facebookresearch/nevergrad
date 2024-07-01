import numpy as np


class EnhancedRefinedAdaptiveDifferentialSpiralSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality as per problem constraints

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = np.zeros(self.dim)

        # Initialize a population around the search space
        population_size = 150  # Further increased population for better exploration
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        # Initialize spiral dynamics with adaptive parameters
        radius = 5.0
        angle_increment = 2 * np.pi / population_size
        evaluations_left = self.budget - population_size
        radius_decay = 0.98  # More gradual decay to maintain exploration longer
        angle_speed_increase = 1.02  # Lower increase rate for more controlled search

        while evaluations_left > 0:
            for i in range(population_size):
                # Select random indices for differential mutation
                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = population[indices[0]], population[indices[1]], population[indices[2]]

                # Perform differential mutation
                mutant = a + 0.8 * (b - c)  # Adjusted differential weight for stability
                mutant = np.clip(mutant, -5.0, 5.0)

                # Adjust the mutant with spiral motion
                angle = angle_increment * i
                offset = radius * np.array([np.cos(angle), np.sin(angle)] + [0] * (self.dim - 2))
                candidate = mutant + offset
                candidate = np.clip(candidate, -5.0, 5.0)

                # Evaluate candidate
                f_candidate = func(candidate)
                evaluations_left -= 1

                # Greedy selection and update optimal solution
                if f_candidate < fitness[i]:
                    population[i] = candidate
                    fitness[i] = f_candidate
                    if f_candidate < self.f_opt:
                        self.f_opt = f_candidate
                        self.x_opt = candidate

            # Adaptive update of spiral parameters
            radius *= radius_decay
            angle_increment *= angle_speed_increase

        return self.f_opt, self.x_opt
