import numpy as np


class UltraRefinedSpiralDifferentialClimberV3:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = np.zeros(self.dim)

        # Initialize population and parameters
        population_size = 600  # Further increased size for exploratory coverage
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        # Parameters for spiral dynamics and enhanced search techniques
        min_radius = 0.005  # Further reduced for finer local search
        max_radius = 5.0  # Adjusted to fully utilize the boundary
        radius_decay = 0.98  # Slower decay to maintain spiral influence longer
        mutation_factor = 0.7  # Further reduced mutation for stability
        crossover_probability = 0.9  # Further increased crossover probability

        # Enhanced gradient-like search parameters
        step_size = 0.02  # Further reduced step size for more precise adjustments
        gradient_steps = 15  # Increased number of gradient steps

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

                # Apply spiral dynamics
                radius = max(min_radius, max_radius * radius_decay ** (self.budget - evaluations_left))
                angle = 2 * np.pi * np.random.rand()
                spiral_offset = radius * np.array([np.cos(angle), np.sin(angle)] + [0] * (self.dim - 2))
                trial += spiral_offset
                trial = np.clip(trial, -5.0, 5.0)

                # Enhanced gradient-like search for local refinement
                for _ in range(gradient_steps):
                    new_trial = trial + np.random.randn(self.dim) * step_size
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
