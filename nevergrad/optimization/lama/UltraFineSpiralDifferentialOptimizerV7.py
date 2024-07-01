import numpy as np


class UltraFineSpiralDifferentialOptimizerV7:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = np.zeros(self.dim)

        # Initialize parameters
        population_size = 150  # Further reduced population for more focus on selected vectors
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        # Spiral and mutation parameters
        min_radius = 0.00001  # Reduced minimum radius for finer local movements
        max_radius = 1.5  # Reduced max radius for tighter local focus
        radius_decay = 0.95  # Slower radius decay for extended local exploration
        mutation_factor = 1.0  # Enhanced mutation factor for aggressive diversity
        crossover_probability = 0.8  # Slightly reduced to allow more mutant characteristics

        # Advanced gradient local search parameters
        step_size = 0.001  # Further refined step size
        gradient_steps = 200  # Increased gradient steps for deeper local optimization

        evaluations_left = self.budget - population_size

        while evaluations_left > 0:
            for i in range(population_size):
                # Differential evolution mutation
                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + mutation_factor * (b - c), -5.0, 5.0)

                # Crossover operation
                cross_points = np.random.rand(self.dim) < crossover_probability
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Spiral dynamic integration with reduced radius and slower decay
                radius = max(min_radius, max_radius * radius_decay ** (self.budget - evaluations_left))
                angle = 2 * np.pi * np.random.rand()
                spiral_offset = radius * np.array([np.cos(angle), np.sin(angle)] + [0] * (self.dim - 2))
                trial += spiral_offset
                trial = np.clip(trial, -5.0, 5.0)

                # Enhanced gradient descent-like local search
                for _ in range(gradient_steps):
                    new_trial = trial + np.random.normal(scale=step_size, size=self.dim)
                    new_trial = np.clip(new_trial, -5.0, 5.0)
                    f_new_trial = func(new_trial)
                    evaluations_left -= 1
                    if evaluations_left <= 0:
                        break
                    if f_new_trial < func(trial):
                        trial = new_trial

                # Evaluation of the new solution
                f_trial = func(trial)
                evaluations_left -= 1
                if evaluations_left <= 0:
                    break

                # Population update
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

        return self.f_opt, self.x_opt
