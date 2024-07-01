import numpy as np


class OptimalBalanceSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Initialization
        current_point = np.random.uniform(-5.0, 5.0, self.dim)
        current_f = func(current_point)
        if current_f < self.f_opt:
            self.f_opt = current_f
            self.x_opt = current_point

        # Configuration for adaptive scaling
        max_scale = 5.0
        min_scale = 0.01
        scale = max_scale
        scale_decay = 0.95  # More gradual decay

        # Configuration for adaptive exploration
        initial_exploration_prob = 0.5
        min_exploration_prob = 0.1
        exploration_decay = 0.99  # Slower decay rate for exploration

        exploration_probability = initial_exploration_prob

        for i in range(1, self.budget):
            if np.random.rand() < exploration_probability:
                # Exploration with more controlled scaling
                scale_range = np.linspace(min_scale, max_scale, num=self.budget)
                candidate = current_point + np.random.uniform(-scale_range[i], scale_range[i], self.dim)
                candidate = np.clip(candidate, -5.0, 5.0)  # Ensure within bounds
            else:
                # Exploitation with adaptive perturbation
                perturbation = np.random.normal(0, scale, self.dim)
                candidate = current_point + perturbation
                candidate = np.clip(candidate, -5.0, 5.0)  # Ensure within bounds

            candidate_f = func(candidate)

            # Update if the candidate is better
            if candidate_f < current_f:
                current_point = candidate
                current_f = candidate_f
                if candidate_f < self.f_opt:
                    self.f_opt = candidate_f
                    self.x_opt = candidate

            # Update exploration probability and scale
            exploration_probability *= exploration_decay
            exploration_probability = max(min_exploration_prob, exploration_probability)
            scale *= scale_decay
            scale = max(min_scale, scale)  # Ensuring scale does not become too small

        return self.f_opt, self.x_opt
