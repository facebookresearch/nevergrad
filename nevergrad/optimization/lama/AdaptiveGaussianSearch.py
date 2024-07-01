import numpy as np


class AdaptiveGaussianSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem

    def __call__(self, func):
        # Initialize variables
        self.f_opt = np.inf
        self.x_opt = None
        # Initial guess at the center of the search space
        current_point = np.random.uniform(-5.0, 5.0, self.dim)
        current_f = func(current_point)

        # Update optimal solution if the initial guess is better
        if current_f < self.f_opt:
            self.f_opt = current_f
            self.x_opt = current_point

        # Set initial scale of the Gaussian perturbations
        scale = 1.0

        # Main optimization loop
        for i in range(self.budget - 1):
            # Generate a new candidate by perturbing the current point
            candidate = current_point + np.random.normal(0, scale, self.dim)
            # Ensure the candidate stays within bounds
            candidate = np.clip(candidate, -5.0, 5.0)
            candidate_f = func(candidate)

            # If the candidate is better, move there and increase the perturbation scale
            if candidate_f < current_f:
                current_point = candidate
                current_f = candidate_f
                scale *= 1.1  # Encourage exploration
                # Update the optimal solution found
                if candidate_f < self.f_opt:
                    self.f_opt = candidate_f
                    self.x_opt = candidate

            # If not better, decrease the perturbation scale to refine search
            else:
                scale *= 0.9  # Encourage exploitation

        return self.f_opt, self.x_opt
