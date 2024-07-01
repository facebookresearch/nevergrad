import numpy as np


class UltraRefinedConvergenceSpiralSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality as per problem's constraints

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Initialize the centroid and search parameters
        centroid = np.random.uniform(-5.0, 5.0, self.dim)
        radius = 5.0  # Initial radius
        angle_increment = np.pi / 16  # Finer initial angle increment for exploration

        # Adaptive decay rates are more responsive to feedback
        radius_decay = 0.90  # Modestly aggressive radius decay
        angle_refinement = 0.95  # Less aggressive angle refinement
        evaluations_left = self.budget
        min_radius = 0.001  # Very fine minimum radius for detailed exploration

        # Dynamic adjustment scales based on the feedback
        optimal_change_factor = 1.85  # Dynamically adjust the decay rates
        no_improvement_count = 0
        last_best_f = np.inf

        # Improved escape mechanism with adaptive triggers
        escape_momentum = 0  # Track escape momentum
        escape_trigger = 10  # Sooner trigger for escape mechanism

        while evaluations_left > 0:
            points = []
            function_values = []
            num_points = max(int(2 * np.pi / angle_increment), 8)  # Ensure sufficient sampling

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
                radius_decay = min(radius_decay * optimal_change_factor, 0.93)  # Slightly less aggressive
                angle_refinement = min(angle_refinement * optimal_change_factor, 0.93)
            else:
                no_improvement_count += 1

            # Update centroid based on feedback
            if points:
                best_index = np.argmin(function_values)
                centroid = points[best_index]

            # Dynamically adjust escape and search parameters
            if no_improvement_count > escape_trigger:
                radius = min(radius / radius_decay, 5.0)  # Increase radius to escape
                angle_increment = np.pi / 8  # Reset angle increment to improve exploration
                no_improvement_count = 0
                escape_momentum += 1
            else:
                radius *= radius_decay  # Tighten search
                radius = max(radius, min_radius)  # Ensure not too small
                angle_increment *= angle_refinement  # Refine search

        return self.f_opt, self.x_opt
