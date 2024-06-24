import numpy as np


class RefinedSpiralSearchOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Initial point in the center of the search space
        initial_point = np.zeros(self.dim)
        current_point = initial_point
        current_f = func(current_point)
        if current_f < self.f_opt:
            self.f_opt = current_f
            self.x_opt = current_point

        # Parameters for spiral movement
        radius = 5.0  # Maximum extent of the search space
        angle_increment = np.pi / 8  # Smaller incremental angle for finer spiral
        radius_decrement_factor = 0.9  # Slightly slower radius reduction to explore more thoroughly
        spiral_budget = self.budget

        while spiral_budget > 0:
            num_points = int(2 * np.pi / angle_increment)
            for i in range(num_points):
                if spiral_budget <= 0:
                    break

                angle = i * angle_increment
                for dim in range(self.dim):  # Spiral in all dimensions
                    dx = radius * np.cos(angle)
                    dy = radius * np.sin(angle)
                    delta = np.zeros(self.dim)
                    delta[dim % self.dim] = dx
                    delta[(dim + 1) % self.dim] = dy

                    candidate_point = current_point + delta
                    candidate_point = np.clip(
                        candidate_point, -5.0, 5.0
                    )  # Ensure the candidate is within bounds
                    candidate_f = func(candidate_point)
                    spiral_budget -= 1

                    if candidate_f < self.f_opt:
                        self.f_opt = candidate_f
                        self.x_opt = candidate_point
                        current_point = candidate_point  # Move spiral center to new best location

            # Reduce the radius for the next spiral cycle
            radius *= radius_decrement_factor

        return self.f_opt, self.x_opt
