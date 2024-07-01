import numpy as np


class EnhancedConvergenceAcceleratedSpiralSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality as per problem's constraints

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Initial setup
        centroid = np.random.uniform(-5.0, 5.0, self.dim)
        radius = 5.0  # Start with a full range
        angle_increment = np.pi / 8  # More precise initial angle for exploration

        # Adaptive parameters
        radius_decay = 0.92  # More gradual radius decay to maintain broader search longer
        angle_refinement = 0.88  # More gradual angle refinement for more thorough search
        evaluations_left = self.budget
        min_radius = 0.0001  # Even finer minimum radius for very detailed exploration

        # Dynamic angle adjustment based on feedback loop
        optimal_change_factor = 1.9  # Slightly less aggressive dynamic adjustment
        no_improvement_count = 0
        last_best_f = np.inf

        # Improved escape mechanism parameters
        escape_momentum = 0  # To track when to increase radius temporarily
        escape_trigger = 15  # Number of cycles without improvement to trigger escape

        while evaluations_left > 0:
            points = []
            function_values = []
            num_points = max(int(2 * np.pi / angle_increment), 6)  # Ensure at least 6 points

            for i in range(num_points):
                if evaluations_left <= 0:
                    break

                angle = i * angle_increment
                displacement = radius * np.array([np.cos(angle), np.sin(angle)] + [0] * (self.dim - 2))
                new_point = centroid + displacement
                new_point = np.clip(new_point, -5.0, 5.0)  # Enforce bounds

                f_val = func(new_point)
                evaluations_left -= 1

                points.append(new_point)
                function_values.append(f_val)

                if f_val < self.f_opt:
                    self.f_opt = f_val
                    self.x_opt = new_point

            # Determine if there has been an improvement
            if self.f_opt < last_best_f:
                last_best_f = self.f_opt
                no_improvement_count = 0
                radius_decay = min(radius_decay * optimal_change_factor, 0.94)
                angle_refinement = min(angle_refinement * optimal_change_factor, 0.94)
            else:
                no_improvement_count += 1

            # Update centroid to new best point
            if points:
                best_index = np.argmin(function_values)
                centroid = points[best_index]

            # Adjust search parameters when stuck
            if no_improvement_count > escape_trigger:
                radius = min(radius / radius_decay, 5.0)  # Increase radius to escape
                angle_increment = np.pi / 4  # Reset angle increment to improve exploration
                no_improvement_count = 0
                escape_momentum += 1

            # Gradual refinement of search
            else:
                radius *= radius_decay  # Tighten search
                radius = max(radius, min_radius)  # Ensure not too small
                angle_increment *= angle_refinement  # Refine search

        return self.f_opt, self.x_opt
