import numpy as np


class IntelligentPerturbationSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem

    def __call__(self, func):
        # Initialize variables
        self.f_opt = np.inf
        self.x_opt = None

        # Start with a random point in the search space
        current_point = np.random.uniform(-5.0, 5.0, self.dim)
        current_f = func(current_point)

        # Update optimal solution if the initial guess is better
        if current_f < self.f_opt:
            self.f_opt = current_f
            self.x_opt = current_point

        # Set initial scale of the Gaussian perturbations
        scale = 0.5
        min_scale = 0.01
        max_scale = 1.0

        # Main optimization loop
        for i in range(1, self.budget):
            # Adjust scale dynamically based on iteration count
            if i < self.budget // 3:
                scale = max_scale
            elif i < 2 * self.budget // 3:
                scale = 0.5 * (max_scale + min_scale)
            else:
                scale = min_scale

            # Generate a new candidate by perturbing the current point
            candidate = current_point + np.random.normal(0, scale, self.dim)
            candidate = np.clip(candidate, -5.0, 5.0)
            candidate_f = func(candidate)

            # Simple acceptance criterion: accept if the candidate is better
            if candidate_f < current_f:
                current_point = candidate
                current_f = candidate_f

                # Update optimal solution found
                if candidate_f < self.f_opt:
                    self.f_opt = candidate_f
                    self.x_opt = candidate

            # Intelligently adjust scale based on progress
            if candidate_f < current_f:
                scale = min(scale * 1.1, max_scale)  # Encourage exploration
            else:
                scale = max(scale * 0.9, min_scale)  # Encourage exploitation

        return self.f_opt, self.x_opt
