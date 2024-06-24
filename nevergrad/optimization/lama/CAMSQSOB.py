import numpy as np


class CAMSQSOB:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5  # Dimensionality of the BBOB test suite
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        last_direction = np.zeros(self.dimension)  # Initialize the last_direction properly

        current_position = np.random.uniform(self.lower_bound, self.upper_bound, self.dimension)
        current_fitness = func(current_position)
        self.update_optimum(current_position, current_fitness)

        delta = 0.5  # Initial step size
        alpha = 0.9  # More aggressive reduction factor for step size
        beta = 0.5  # Momentum term for stabilizing the direction update
        iteration = 1

        while iteration < self.budget:
            scale_changes = [delta * (0.5**i) for i in range(3)]
            for scale in scale_changes:
                if iteration >= self.budget:
                    break
                points, fitnesses = self.generate_points(func, current_position, scale)
                A, b = self.fit_quadratic(points, fitnesses)

                if np.linalg.cond(A) < 1e10:  # Condition to check invertibility
                    step_direction = -np.linalg.solve(A, b)
                    direction = beta * last_direction + (1 - beta) * step_direction
                    new_position = np.clip(current_position + direction, self.lower_bound, self.upper_bound)
                    new_fitness = func(new_position)
                    self.update_optimum(new_position, new_fitness)

                    # Opposition-based Learning
                    opposite_position = self.lower_bound + self.upper_bound - new_position
                    opposite_fitness = func(opposite_position)
                    self.update_optimum(opposite_position, opposite_fitness)

                    if opposite_fitness < new_fitness:
                        new_position, new_fitness = opposite_position, opposite_fitness

                    if new_fitness < current_fitness:
                        current_position, current_fitness = new_position, new_fitness
                        delta = min(delta / alpha, 1.0)  # Adjust delta upon improvement
                        last_direction = direction
                    else:
                        delta *= alpha  # Reduce delta upon failure

                iteration += 2 * self.dimension + 2

        return self.f_opt, self.x_opt

    def generate_points(self, func, center, delta):
        points = np.vstack(
            [center + delta * np.eye(self.dimension)[i] for i in range(self.dimension)]
            + [center - delta * np.eye(self.dimension)[i] for i in range(self.dimension)]
        )
        fitnesses = np.array([func(point) for point in points])
        return points, fitnesses

    def update_optimum(self, x, f):
        if f < self.f_opt:
            self.f_opt = f
            self.x_opt = x

    def fit_quadratic(self, points, fitnesses):
        X = np.hstack([np.ones((len(fitnesses), 1)), points])
        coeffs = np.linalg.lstsq(X, fitnesses - fitnesses.min(), rcond=None)[0]
        A = np.diag(coeffs[1:])
        b = coeffs[: self.dimension]
        return A, b
