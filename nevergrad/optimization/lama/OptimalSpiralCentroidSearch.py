import numpy as np


class OptimalSpiralCentroidSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is set as per the problem's constraints

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Initialize the centroid position of the search space
        centroid = np.zeros(self.dim)
        radius = 5.0  # Initial radius correlating to the full search space
        angle_increment = np.pi / 8  # Initial angle increment for broad exploration

        # Parameters for adapting the search
        radius_decay = 0.98  # Decrement for radius to focus search progressively
        angle_refinement = 0.95  # Refinement of angle increment for increased precision
        evaluations_left = self.budget
        min_radius = 0.05  # Minimum radius to maintain a level of exploration

        while evaluations_left > 0:
            points = []
            function_values = []
            num_points = int(2 * np.pi / angle_increment)

            for i in range(num_points):
                if evaluations_left <= 0:
                    break

                angle = i * angle_increment
                new_point = centroid + radius * np.array(
                    [np.cos(angle), np.sin(angle)] + [0] * (self.dim - 2)
                )
                new_point = np.clip(new_point, -5.0, 5.0)  # Ensure the candidate is within bounds

                f_val = func(new_point)
                evaluations_left -= 1

                points.append(new_point)
                function_values.append(f_val)

                if f_val < self.f_opt:
                    self.f_opt = f_val
                    self.x_opt = new_point

            # Update the centroid to the best found point in the current iteration
            if points:
                best_index = np.argmin(function_values)
                centroid = points[best_index]

            # Reduce the radius and refine the angle increment for more focused search
            radius *= radius_decay
            radius = max(radius, min_radius)  # Avoid too small radius to prevent stagnation
            angle_increment *= angle_refinement

        return self.f_opt, self.x_opt
