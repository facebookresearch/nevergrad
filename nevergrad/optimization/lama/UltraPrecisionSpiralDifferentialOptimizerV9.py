import numpy as np


class UltraPrecisionSpiralDifferentialOptimizerV9:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality set as constant

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = np.zeros(self.dim)

        # Initialize population
        population_size = 50  # Smaller population for faster convergence
        population = np.random.uniform(-5.0, 5.0, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        # Tuned parameters for spirals and mutation
        min_radius = 0.00005  # Further reduced minimum radius for ultra-precision
        max_radius = 0.5  # Reduced maximum radius to focus nearer to current best regions
        radius_decay = 0.99  # Slower decay rate to maintain explorative behavior longer
        mutation_factor = 0.8  # Lower mutation factor to refine search rather than diversify too much
        crossover_probability = 0.8  # Higher crossover probability to ensure thorough exploration

        # Local search adjustments
        step_size = 0.0001  # Smaller step size for ultra-fine tuning
        gradient_steps = 50  # Fewer steps to save budget for more global exploration

        evaluations_left = self.budget - population_size

        while evaluations_left > 0:
            for i in range(population_size):
                # Differential evolution mutation strategy
                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = population[indices]
                mutant = np.clip(a + mutation_factor * (b - c), -5.0, 5.0)

                # Crossover operation
                cross_points = np.random.rand(self.dim) < crossover_probability
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Spiral motion integration for detailed search
                radius = max(min_radius, max_radius * radius_decay ** (self.budget - evaluations_left))
                angle = 2 * np.pi * np.random.rand()
                spiral_offset = radius * np.array([np.cos(angle), np.sin(angle)] + [0] * (self.dim - 2))
                trial += spiral_offset
                trial = np.clip(trial, -5.0, 5.0)

                # Local search with finer steps
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
