import numpy as np


class QuantumSimulatedAnnealing:
    def __init__(self, budget=10000, initial_temp=1.0, cooling_rate=0.999):
        self.budget = budget
        self.dim = 5
        self.temp = initial_temp
        self.cooling_rate = cooling_rate

    def _quantum_step(self, x):
        return x + np.random.uniform(-0.1 * self.temp, 0.1 * self.temp, size=self.dim)

    def _acceptance_probability(self, candidate_f, current_f):
        return np.exp((current_f - candidate_f) / self.temp)

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        current_x = np.random.uniform(-5.0, 5.0, size=self.dim)
        current_f = func(current_x)

        for i in range(self.budget):
            candidate_x = self._quantum_step(current_x)
            candidate_x = np.clip(candidate_x, -5.0, 5.0)
            candidate_f = func(candidate_x)

            if candidate_f < current_f or np.random.rand() < self._acceptance_probability(
                candidate_f, current_f
            ):
                current_x = candidate_x
                current_f = candidate_f

            if current_f < self.f_opt:
                self.f_opt = current_f
                self.x_opt = current_x

            self.temp *= self.cooling_rate

        return self.f_opt, self.x_opt
