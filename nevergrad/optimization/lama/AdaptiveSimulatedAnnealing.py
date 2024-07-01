import numpy as np


class AdaptiveSimulatedAnnealing:
    def __init__(self, budget=10000, initial_temp=10.0, cooling_rate=0.95, min_temp=1e-5):
        self.budget = budget
        self.dim = 5
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp

    def acceptance_probability(self, energy, new_energy, temp):
        if new_energy < energy:
            return 1.0
        return np.exp((energy - new_energy) / temp)

    def __call__(self, func):
        current_state = np.random.uniform(-5.0, 5.0, self.dim)
        best_state = current_state
        current_energy = func(current_state)
        best_energy = current_energy
        temp = self.initial_temp

        while temp > self.min_temp:
            for _ in range(self.budget):
                new_state = current_state + np.random.normal(0, 1, self.dim)
                new_state = np.clip(new_state, -5.0, 5.0)
                new_energy = func(new_state)
                ap = self.acceptance_probability(current_energy, new_energy, temp)
                if ap > np.random.rand():
                    current_state = new_state
                    current_energy = new_energy
                    if new_energy < best_energy:
                        best_state = new_state
                        best_energy = new_energy
            temp *= self.cooling_rate

        return best_energy, best_state
