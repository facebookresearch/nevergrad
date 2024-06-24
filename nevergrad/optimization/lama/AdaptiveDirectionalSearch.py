import numpy as np


class AdaptiveDirectionalSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = None

    def __call__(self, func):
        self.dim = len(func.bounds.lb)
        self.f_opt = np.Inf
        self.x_opt = None
        self.alpha = 0.1  # Step size
        self.beta = 0.5  # Contraction factor
        self.gamma = 2.0  # Expansion factor
        self.delta = 1e-5  # Small perturbation to avoid stagnation

        x = np.random.uniform(func.bounds.lb, func.bounds.ub)  # Initial solution
        f = func(x)
        evaluations = 1

        while evaluations < self.budget:
            direction = np.random.randn(self.dim)
            direction /= np.linalg.norm(direction)  # Normalize direction vector

            # Try expanding
            x_new = x + self.gamma * self.alpha * direction
            x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)
            f_new = func(x_new)
            evaluations += 1

            if f_new < f:
                x = x_new
                f = f_new
                self.alpha *= self.gamma
            else:
                # Try contracting
                x_new = x + self.beta * self.alpha * direction
                x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)
                f_new = func(x_new)
                evaluations += 1

                if f_new < f:
                    x = x_new
                    f = f_new
                    self.alpha *= self.beta
                else:
                    # Apply small perturbation to avoid getting stuck
                    x_new = x + self.delta * direction
                    x_new = np.clip(x_new, func.bounds.lb, func.bounds.ub)
                    f_new = func(x_new)
                    evaluations += 1

                    if f_new < f:
                        x = x_new
                        f = f_new

            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x

        return self.f_opt, self.x_opt
