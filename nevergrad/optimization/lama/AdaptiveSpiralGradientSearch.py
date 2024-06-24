import numpy as np


class AdaptiveSpiralGradientSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality as per problem's constraints

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Initial setup
        centroid = np.random.uniform(-5.0, 5.0, self.dim)
        radius = 5.0  # Start with a full range
        angle_increment = np.pi / 4  # Broader angle for initial exploration

        # Adaptive parameters
        radius_decay = 0.95  # Slowly decrease radius
        angle_refinement = 0.90  # Refine angles for closer exploration
        evaluations_left = self.budget
        min_radius = 0.01  # Prevent the radius from becoming too small

        # This array holds the last few best points to calculate a moving centroid
        historical_best = centroid.copy()

        while evaluations_left > 0:
            points = []
            function_values = []
            num_points = max(int(2 * np.pi / angle_increment), 3)  # Ensure at least 3 points

            for i in range(num_points):
                if evaluations_left <= 0:
                    break

                angle = i * angle_increment
                for offset in np.linspace(0, 2 * np.pi, num_points, endpoint=False):
                    displacement = radius * np.array(
                        [np.cos(angle + offset), np.sin(angle + offset)] + [0] * (self.dim - 2)
                    )
                    new_point = centroid + displacement
                    new_point = np.clip(new_point, -5.0, 5.0)  # Enforce bounds

                    f_val = func(new_point)
                    evaluations_left -= 1

                    points.append(new_point)
                    function_values.append(f_val)

                    if f_val < self.f_opt:
                        self.f_opt = f_val
                        self.x_opt = new_point

            # Update the centroid towards the best found point in this iteration
            if points:
                best_index = np.argmin(function_values)
                historical_best = 0.8 * historical_best + 0.2 * points[best_index]
                centroid = historical_best

            # Dynamically update radius and angle increment
            radius *= radius_decay
            radius = max(radius, min_radius)
            angle_increment *= angle_refinement

        return self.f_opt, self.x_opt
