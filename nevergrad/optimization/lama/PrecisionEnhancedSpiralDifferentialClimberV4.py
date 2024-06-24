import numpy as np


class PrecisionEnhancedSpiralDifferentialClimberV4:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = np.zeros(self.dim)

        # Initialize population and parameters
        population_size = 500  # Adjust population size for better performance
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        # Adjust parameters for spiral dynamics and evolutionary strategies
        min_radius = 0.001  # Further refinement for local search
        max_radius = 5.0  # Maintaining the boundary limit
        radius_decay = 0.99  # Slower decay for extended influence of spiral movement
        mutation_factor = 0.6  # Reduced mutation for controlled exploration
        crossover_probability = 0.92  # Slightly increased for enhanced mixing

        # Advanced gradient-like search parameters
        step_size = 0.01  # Reduced step size for high precision
        gradient_steps = 20  # Increased steps for exhaustive local search

        evaluations_left = self.budget - population_size

        while evaluations_left > 0:
            for i in range(population_size):
                # Differential Evolution Strategy
                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + mutation_factor * (b - c), -5.0, 5.0)

                # Crossover
                cross_points = np.random.rand(self.dim) < crossover_probability
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Spiral dynamics integration
                radius = max(min_radius, max_radius * radius_decay ** (self.budget - evaluations_left))
                angle = 2 * np.pi * np.random.rand()
                spiral_offset = radius * np.array([np.cos(angle), np.sin(angle)] + [0] * (self.dim - 2))
                trial += spiral_offset
                trial = np.clip(trial, -5.0, 5.0)

                # Advanced gradient-like search for fine-tuning
                for _ in range(gradient_steps):
                    new_trial = trial + np.random.normal(scale=step_size, size=self.dim)
                    new_trial = np.clip(new_trial, -5.0, 5.0)
                    f_new_trial = func(new_trial)
                    evaluations_left -= 1
                    if f_new_trial < func(trial):
                        trial = new_trial

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
