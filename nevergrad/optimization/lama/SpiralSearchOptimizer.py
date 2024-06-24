import numpy as np


class SpiralSearchOptimizer:
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
        angle_increment = np.pi / 4  # Incremental angle for spiral
        radius_decrement_factor = 0.95  # Reduce radius after each full spiral
        spiral_budget = self.budget

        while spiral_budget > 0:
            num_points = int(2 * np.pi / angle_increment)
            for i in range(num_points):
                if spiral_budget <= 0:
                    break

                angle = i * angle_increment
                dx = radius * np.cos(angle)
                dy = radius * np.sin(angle)
                candidate_point = current_point + np.array(
                    [dx, dy, 0, 0, 0]
                )  # Spiral in 2D, constant in other dimensions
                candidate_point = np.clip(candidate_point, -5.0, 5.0)  # Ensure the candidate is within bounds

                candidate_f = func(candidate_point)
                spiral_budget -= 1

                if candidate_f < self.f_opt:
                    self.f_opt = candidate_f
                    self.x_opt = candidate_point
                    current_point = candidate_point  # Move spiral center to new best location

            # Reduce the radius for the next spiral cycle
            radius *= radius_decrement_factor

        return self.f_opt, self.x_opt
