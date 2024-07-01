import numpy as np


class AdvancedRefinedSpiralSearchOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Initial point in the center of the search space
        initial_point = np.random.uniform(-5.0, 5.0, self.dim)
        current_point = initial_point
        current_f = func(current_point)
        if current_f < self.f_opt:
            self.f_opt = current_f
            self.x_opt = current_point

        # Parameters for spiral movement
        radius = 5.0  # Maximum extent of the search space
        angle_increment = np.pi / 12  # Reduced angle increment for finer detail
        radius_decrement_factor = 0.95  # Slower radius reduction for thorough exploration
        spiral_budget = self.budget

        # Adaptive refinement based on previous results
        adaptive_radius_change = 0.1  # Gradual increase in radius adjustment
        min_radius = 0.1  # Minimum radius to prevent infinitesimal spirals

        while spiral_budget > 0:
            num_points = int(2 * np.pi / angle_increment)
            for i in range(num_points):
                if spiral_budget <= 0:
                    break

                angle = i * angle_increment
                candidate_point = current_point.copy()
                radius *= 1.0 + adaptive_radius_change  # Dynamically increase radius
                radius = max(radius, min_radius)  # Maintain a minimum radius

                for dim in range(self.dim):  # Create a more complex spiral
                    dx = radius * np.cos(angle + 2 * np.pi * dim / self.dim)
                    dy = radius * np.sin(angle + 2 * np.pi * dim / self.dim)
                    candidate_point[dim % self.dim] += dx
                    candidate_point[(dim + 1) % self.dim] += dy

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
            angle_increment *= 0.99  # Gradually refine the angle increment for more precise spiral turns

        return self.f_opt, self.x_opt
