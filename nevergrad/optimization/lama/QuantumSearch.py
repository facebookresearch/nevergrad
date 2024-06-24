import numpy as np


class QuantumSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.alpha = 0.1  # Step size parameter

    def _quantum_step(self, x):
        return x + np.random.uniform(-self.alpha, self.alpha, size=self.dim)

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        x = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))

        for i in range(self.budget):
            x[i] = self._quantum_step(x[i])
            x[i] = np.clip(x[i], -5.0, 5.0)  # Ensure within bounds

            f = func(x[i])
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x[i]

        return self.f_opt, self.x_opt
