import numpy as np


class UltraOptimizedSpiralDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = np.zeros(self.dim)

        # Initialize population and parameters
        population_size = 200  # Increased size for broader initial coverage
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        min_radius = 0.1  # Minimum radius for spiral
        max_radius = 5.0  # Starting radius for spiral
        radius_decay = 0.97  # More aggressive decay
        mutation_factor = 0.9  # Higher mutation for aggressive exploration
        crossover_probability = 0.7  # Probability for crossover

        evaluations_left = self.budget - population_size

        while evaluations_left > 0:
            for i in range(population_size):
                # Differential Evolution Strategy
                a, b, c = np.random.choice(population_size, 3, replace=False)
                mutant = population[a] + mutation_factor * (population[b] - population[c])
                mutant = np.clip(mutant, -5.0, 5.0)

                # Crossover
                cross_points = np.random.rand(self.dim) < crossover_probability
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Spiral adjustment
                radius = max(min_radius, max_radius * radius_decay ** (self.budget - evaluations_left))
                angle = 2 * np.pi * i / population_size
                spiral_offset = radius * np.array([np.cos(angle), np.sin(angle)] + [0] * (self.dim - 2))
                trial += spiral_offset
                trial = np.clip(trial, -5.0, 5.0)

                # Evaluation
                f_trial = func(trial)
                evaluations_left -= 1

                # Selection
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

        return self.f_opt, self.x_opt
