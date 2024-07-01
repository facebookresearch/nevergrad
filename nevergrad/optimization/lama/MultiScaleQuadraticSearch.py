import numpy as np


class MultiScaleQuadraticSearch:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5  # Dimensionality of the BBOB test suite
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Initial random position
        current_position = np.random.uniform(self.lower_bound, self.upper_bound, self.dimension)
        current_fitness = func(current_position)
        self.update_optimum(current_position, current_fitness)

        # Control parameters
        delta = 0.5  # Initial step size
        alpha = 0.5  # Reduction factor for step size
        beta = 0.3  # Momentum term
        last_direction = np.zeros(self.dimension)
        epsilon = 1e-6  # Convergence criterion

        iteration = 1
        while iteration < self.budget:
            scales = [delta * (0.5**i) for i in range(3)]  # Different scales for exploration
            for scale in scales:
                if iteration >= self.budget:
                    break
                points, fitnesses = self.generate_points(func, current_position, scale)
                A, b = self.fit_quadratic(current_position, points, fitnesses)

                if np.linalg.cond(A) < 1 / epsilon:
                    try:
                        step_direction = -np.linalg.inv(A).dot(b)
                        direction = beta * last_direction + (1 - beta) * step_direction
                        new_position = current_position + direction
                        new_position = np.clip(new_position, self.lower_bound, self.upper_bound)
                        new_fitness = func(new_position)
                    except np.linalg.LinAlgError:
                        continue

                    if new_fitness < current_fitness:
                        current_position, current_fitness = new_position, new_fitness
                        last_direction = direction
                        self.update_optimum(current_position, current_fitness)
                        delta = min(delta / alpha, 1.0)  # Adjust delta upon improvement
                    else:
                        delta *= alpha  # Reduce delta upon failure

                iteration += 2 * self.dimension + 1

        return self.f_opt, self.x_opt

    def generate_points(self, func, center, delta):
        points = np.array(
            [center + delta * np.eye(self.dimension)[:, i] for i in range(self.dimension)]
            + [center - delta * np.eye(self.dimension)[:, i] for i in range(self.dimension)]
        )
        points = np.clip(points, self.lower_bound, self.upper_bound)
        fitnesses = np.array([func(point) for point in points])
        return points, fitnesses

    def update_optimum(self, x, f):
        if f < self.f_opt:
            self.f_opt = f
            self.x_opt = x

    def fit_quadratic(self, center, points, fitnesses):
        n = len(points)
        X = np.hstack([np.ones((n, 1)), points - center, ((points - center) ** 2)])
        coeffs = np.linalg.lstsq(X, fitnesses, rcond=None)[0]
        A = np.diag(coeffs[1 + self.dimension :])
        b = coeffs[1 : 1 + self.dimension]
        return A, b
