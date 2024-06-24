import numpy as np


class ProgressiveRefinementSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Initializations
        current_point = np.random.uniform(-5.0, 5.0, self.dim)
        current_f = func(current_point)
        if current_f < self.f_opt:
            self.f_opt = current_f
            self.x_opt = current_point

        # Define scales for search
        max_scale = 2.0
        min_scale = 0.001
        scale = max_scale
        scale_decay = 0.98  # Gradually decrease scale

        # Exploration and exploitation configurations
        exploration_probability = 0.3  # Initial probability of exploration
        exploit_prob_growth = 0.005  # Growth rate of the exploitation probability

        for i in range(1, self.budget):
            if np.random.rand() < exploration_probability:
                # Exploration with random point within the search space
                candidate = np.random.uniform(-5.0, 5.0, self.dim)
            else:
                # Exploitation by perturbing the current best point
                perturbation = np.random.normal(0, scale, self.dim)
                candidate = current_point + perturbation
                candidate = np.clip(candidate, -5.0, 5.0)  # Ensure within bounds

            candidate_f = func(candidate)

            # Update current point if the candidate is better
            if candidate_f < current_f:
                current_point = candidate
                current_f = candidate_f
                # Update best found solution
                if candidate_f < self.f_opt:
                    self.f_opt = candidate_f
                    self.x_opt = candidate

            # Adjust exploration-exploitation balance
            exploration_probability = max(0, exploration_probability - exploit_prob_growth)
            exploration_probability = min(1, exploration_probability)

            # Reduce scale to refine search over time
            scale *= scale_decay
            scale = max(min_scale, scale)  # Avoid scale becoming too small

        return self.f_opt, self.x_opt
