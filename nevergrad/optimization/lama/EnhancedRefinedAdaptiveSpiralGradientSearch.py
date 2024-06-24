import numpy as np


class EnhancedRefinedAdaptiveSpiralGradientSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality as per problem's constraints

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Initial setup
        centroid = np.random.uniform(-5.0, 5.0, self.dim)
        radius = 5.0  # Start with a full range
        angle_increment = np.pi / 4  # Initial broad angle for exploration

        # Adaptive parameters
        radius_decay = 0.95  # Gradual decrease radius to extend exploration
        angle_refinement = 0.95  # Slow angle refinement for detailed exploration
        evaluations_left = self.budget
        min_radius = 0.001  # Further reduce min radius for finer precision

        # Maintain a history of the previous best centroids to adjust search dynamics
        historical_best = np.array([centroid.copy() for _ in range(3)])

        while evaluations_left > 0:
            points = []
            function_values = []
            num_points = max(
                int(2 * np.pi / angle_increment), 8
            )  # Ensure at least 8 points for thorough coverage

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

            # Update centroid by averaging the current and historical best points
            if points:
                best_index = np.argmin(function_values)
                historical_best = np.roll(historical_best, -1, axis=0)
                historical_best[-1] = points[best_index]

                # Move the centroid towards the average of historical best locations
                centroid = np.mean(historical_best, axis=0)

            # Dynamically update radius and angle increment
            radius *= radius_decay
            radius = max(radius, min_radius)
            angle_increment *= angle_refinement

        return self.f_opt, self.x_opt
