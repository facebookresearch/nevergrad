import numpy as np


class OctopusSwarmAlgorithm:
    def __init__(self, budget=1000, num_octopuses=20, num_dimensions=5, alpha=0.1, beta=0.5):
        self.budget = budget
        self.num_octopuses = num_octopuses
        self.num_dimensions = num_dimensions
        self.alpha = alpha
        self.beta = beta

    def initialize_positions(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, size=(self.num_octopuses, self.num_dimensions))

    def levy_flight(self):
        sigma = 1.0
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, 1)
        step = u / abs(v) ** (1.5)
        return step

    def move_octopus(self, current_position, best_position, bounds):
        step = self.alpha * (best_position - current_position) + self.beta * self.levy_flight()
        new_position = current_position + step
        new_position = np.clip(new_position, bounds.lb, bounds.ub)
        return new_position

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        bounds = func.bounds

        positions = self.initialize_positions(bounds)
        best_position = positions[0].copy()

        for _ in range(self.budget):
            for i in range(self.num_octopuses):
                new_position = self.move_octopus(positions[i], best_position, bounds)
                f_new = func(new_position)
                f_current = func(positions[i])

                if f_new < f_current:
                    positions[i] = new_position
                    if f_new < func(best_position):
                        best_position = new_position.copy()

        self.f_opt = func(best_position)
        self.x_opt = best_position

        return self.f_opt, self.x_opt
