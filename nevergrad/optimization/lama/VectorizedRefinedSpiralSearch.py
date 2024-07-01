import numpy as np


class VectorizedRefinedSpiralSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality as per problem constraints

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Central point of the search space
        centroid = np.random.uniform(-5.0, 5.0, self.dim)
        radius = 5.0  # Maximum initial radius
        angle_increment = 2 * np.pi / 100  # Angle increment for spiral movement
        evaluations_left = self.budget

        # Adaptive parameters
        radius_decay = 0.97  # Decay radius to focus search over time
        angle_speed_increase = 1.02  # Increase angle speed to cover more area

        while evaluations_left > 0:
            points = []
            # Generate points in a spiral around the centroid
            for _ in range(min(evaluations_left, 100)):
                angle = np.random.uniform(0, 2 * np.pi)
                offset = radius * np.array([np.cos(angle), np.sin(angle)] + [0] * (self.dim - 2))
                point = centroid + offset
                point = np.clip(point, -5.0, 5.0)
                points.append(point)

            points = np.array(points)
            evaluations_left -= len(points)

            # Evaluate the function at all points generated
            results = np.array([func(p) for p in points])

            # Find the best result
            best_idx = np.argmin(results)
            if results[best_idx] < self.f_opt:
                self.f_opt = results[best_idx]
                self.x_opt = points[best_idx]

            # Update the centroid towards the best found point in this iteration
            centroid = self.x_opt

            # Adapt search parameters
            radius *= radius_decay
            angle_increment *= angle_speed_increase

        return self.f_opt, self.x_opt
