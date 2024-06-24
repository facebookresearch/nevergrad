import numpy as np


class MultiStageAdaptiveSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Start with a random point in the search space
        current_point = np.random.uniform(-5.0, 5.0, self.dim)
        current_f = func(current_point)

        # Update if the initial guess is better
        if current_f < self.f_opt:
            self.f_opt = current_f
            self.x_opt = current_point

        # Set initial scale and learning rate
        scale = 0.5
        learning_rate = 0.1

        # Adaptive scale factors
        exploration_scale_factor = 1.2
        exploitation_scale_factor = 0.85

        # Temperature for simulated annealing like probability acceptance
        temperature = 1.0
        min_temperature = 0.01
        temperature_decay = 0.99

        for i in range(1, self.budget):
            # Calculate current temperature
            temperature = max(min_temperature, temperature * temperature_decay)

            # Generate a new candidate by perturbing the current point
            candidate = current_point + np.random.normal(0, scale, self.dim)
            candidate = np.clip(candidate, -5.0, 5.0)
            candidate_f = func(candidate)

            # Calculate acceptance probability using a simulated annealing approach
            if candidate_f < current_f or np.exp((current_f - candidate_f) / temperature) > np.random.rand():
                current_point = candidate
                current_f = candidate_f

                # Update optimal solution found
                if candidate_f < self.f_opt:
                    self.f_opt = candidate_f
                    self.x_opt = candidate

                # Dynamic scale adjustment
                scale *= exploration_scale_factor
            else:
                scale *= exploitation_scale_factor

            # Clamp the scale to prevent it from becoming too large or too small
            scale = np.clip(scale, 0.01, 1.0)

        return self.f_opt, self.x_opt
