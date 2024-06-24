import numpy as np


class DualAdaptiveSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Initial random point
        current_point = np.random.uniform(-5.0, 5.0, self.dim)
        current_f = func(current_point)
        if current_f < self.f_opt:
            self.f_opt = current_f
            self.x_opt = current_point

        # Adaptive parameters
        max_scale = 2.0
        min_scale = 0.001
        scale = max_scale
        scale_decay_rate = 0.98
        long_term_memory = current_point.copy()
        long_term_f = current_f

        # Dual strategy: balance exploration and exploitation
        for i in range(1, self.budget):
            # Decay the scale
            scale *= scale_decay_rate
            scale = max(min_scale, scale)

            # Exploration with a probability that decreases over time
            if np.random.rand() < 0.5 * (1 - i / self.budget):
                # Random exploration within the whole range
                candidate = np.random.uniform(-5.0, 5.0, self.dim)
            else:
                # Exploitation around best known position with decreasing perturbation
                perturbation = np.random.normal(0, scale, self.dim)
                candidate = long_term_memory + perturbation
                candidate = np.clip(candidate, -5.0, 5.0)

            candidate_f = func(candidate)

            # Update if the candidate is better than the current best
            if candidate_f < long_term_f:
                long_term_memory = candidate
                long_term_f = candidate_f
                if candidate_f < self.f_opt:
                    self.f_opt = candidate_f
                    self.x_opt = candidate

        return self.f_opt, self.x_opt
