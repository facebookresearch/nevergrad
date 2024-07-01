import numpy as np


class AdaptiveStepSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # as given in the problem statement
        self.bounds = (-5.0, 5.0)

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        self.step_size = [(self.bounds[1] - self.bounds[0]) / 10] * self.dim  # Initial step size

        # Random initialization
        x = np.random.uniform(self.bounds[0], self.bounds[1], self.dim)
        fx = func(x)
        if fx < self.f_opt:
            self.f_opt = fx
            self.x_opt = x

        evaluations = 1

        while evaluations < self.budget:
            for d in range(self.dim):
                for direction in [-1, 1]:
                    step = np.zeros(self.dim)
                    step[d] = direction * self.step_size[d]
                    x_new = x + step

                    # Ensure the new solution is within bounds
                    x_new = np.clip(x_new, self.bounds[0], self.bounds[1])

                    fx_new = func(x_new)
                    evaluations += 1

                    if fx_new < self.f_opt:
                        self.f_opt = fx_new
                        self.x_opt = x_new
                        x = x_new  # Move to the new position

                    if evaluations >= self.budget:
                        break
                if evaluations >= self.budget:
                    break

            # Adaptively reduce step size
            self.step_size = [s * 0.9 for s in self.step_size]

        return self.f_opt, self.x_opt
