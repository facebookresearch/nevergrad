import numpy as np


class HyperSpiralDifferentialClimber:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = np.zeros(self.dim)

        # Initialize population and parameters
        population_size = 300  # Increased size for broader initial coverage
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        # Parameters for spiral dynamics
        min_radius = 0.05  # Smaller minimum radius for finer local search
        max_radius = 4.5  # Adjusted starting radius
        radius_decay = 0.95  # Slower decay for prolonged spiral influence
        mutation_factor = 0.8  # Adjusted mutation for better local exploitation
        crossover_probability = 0.8  # Increased crossover for more recombination

        # Additional gradient-like update
        step_size = 0.1
        gradient_steps = 5  # Number of gradient-like steps to perform

        evaluations_left = self.budget - population_size

        while evaluations_left > 0:
            for i in range(population_size):
                # Differential Evolution Strategy
                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = a + mutation_factor * (b - c)
                mutant = np.clip(mutant, -5.0, 5.0)

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

                # Additional gradient-like search for local improvement
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
