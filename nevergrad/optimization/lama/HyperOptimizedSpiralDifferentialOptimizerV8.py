import numpy as np


class HyperOptimizedSpiralDifferentialOptimizerV8:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality set as constant

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = np.zeros(self.dim)

        # Initialize population
        population_size = 100  # Adjusted population size for balance between exploration and exploitation
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        # Tuned parameters for spirals and mutation
        min_radius = 0.0001  # More precise minimum radius
        max_radius = 1.0  # Lower maximum radius to focus search
        radius_decay = 0.97  # More gradual decay
        mutation_factor = 1.2  # Increased mutation factor for added diversity
        crossover_probability = 0.7  # Increased crossover rate for more exploration

        # Enhanced local search parameters
        step_size = 0.0005  # Reduced step size for finer local movements
        gradient_steps = 300  # More localized optimization steps

        evaluations_left = self.budget - population_size

        while evaluations_left > 0:
            for i in range(population_size):
                # Differential evolution mutation strategy
                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + mutation_factor * (b - c), -5.0, 5.0)

                # Conduct crossover
                cross_points = np.random.rand(self.dim) < crossover_probability
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Spiral motion integration for non-linear exploration
                radius = max(min_radius, max_radius * radius_decay ** (self.budget - evaluations_left))
                angle = 2 * np.pi * np.random.rand()
                spiral_offset = radius * np.array([np.cos(angle), np.sin(angle)] + [0] * (self.dim - 2))
                trial += spiral_offset
                trial = np.clip(trial, -5.0, 5.0)

                # Perform local search with a gradient descent-like approach
                for _ in range(gradient_steps):
                    new_trial = trial + np.random.normal(scale=step_size, size=self.dim)
                    new_trial = np.clip(new_trial, -5.0, 5.0)
                    f_new_trial = func(new_trial)
                    evaluations_left -= 1
                    if evaluations_left <= 0:
                        break
                    if f_new_trial < func(trial):
                        trial = new_trial

                # Evaluate and update the population
                f_trial = func(trial)
                evaluations_left -= 1
                if evaluations_left <= 0:
                    break

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

        return self.f_opt, self.x_opt
