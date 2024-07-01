import numpy as np


class EnhancedHarmonicLevyDolphinOptimization:
    def __init__(
        self, budget=1000, num_dolphins=20, num_dimensions=5, alpha=0.1, beta=0.5, gamma=0.1, delta=0.2
    ):
        self.budget = budget
        self.num_dolphins = num_dolphins
        self.num_dimensions = num_dimensions
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def initialize_positions(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, size=(self.num_dolphins, self.num_dimensions))

    def levy_flight(self):
        sigma = 1.0
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, 1)
        step = u / abs(v) ** (1.5)
        return step

    def move_dolphin(self, current_position, best_position, previous_best_position, bounds):
        step = (
            self.alpha * (best_position - current_position)
            + self.beta * (previous_best_position - current_position)
            + self.gamma * self.levy_flight()
        )
        new_position = current_position + step
        new_position = np.clip(new_position, bounds.lb, bounds.ub)
        return new_position

    def update_parameters(self, iteration):
        self.alpha = max(0.01, self.alpha * (1 - 0.9 * iteration / self.budget))
        self.beta = min(0.9, self.beta + 0.1 * iteration / self.budget)
        self.gamma = max(0.01, self.gamma * (1 - 0.8 * iteration / self.budget))

    def levy_harmonic_search(self, func, position):
        new_position = position + self.levy_flight()
        new_position = np.clip(new_position, func.bounds.lb, func.bounds.ub)
        if func(new_position) < func(position):
            return new_position
        else:
            return position

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        bounds = func.bounds

        positions = self.initialize_positions(bounds)
        best_position = positions[0].copy()
        previous_best_position = best_position.copy()

        for i in range(self.budget):
            self.update_parameters(i)
            for j in range(self.num_dolphins):
                new_position = self.move_dolphin(positions[j], best_position, previous_best_position, bounds)
                new_position = self.levy_harmonic_search(func, new_position)

                if func(new_position) < func(positions[j]):
                    positions[j] = new_position
                    if func(new_position) < func(best_position):
                        best_position = new_position.copy()

            previous_best_position = best_position

        self.f_opt = func(best_position)
        self.x_opt = best_position

        return self.f_opt, self.x_opt
