import numpy as np


class EnhancedAdaptiveHarmonySearchWithEnhancedLevyFlightV12:
    def __init__(self, budget, harmony_memory_size=20, global_best_rate=0.1, levy_step_size=0.001):
        self.budget = budget
        self.harmony_memory_size = harmony_memory_size
        self.global_best_rate = global_best_rate
        self.levy_step_size = levy_step_size
        self.convergence_curve = []

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        harmony_memory = np.random.uniform(
            func.bounds.lb, func.bounds.ub, size=(self.harmony_memory_size, len(func.bounds.lb))
        )

        for t in range(self.budget):
            new_harmony = self.generate_new_harmony(harmony_memory, func, t)
            new_harmony_fitness = np.array([func(x) for x in new_harmony])

            min_index = np.argmin(new_harmony_fitness)
            if new_harmony_fitness[min_index] < self.f_opt:
                self.f_opt = new_harmony_fitness[min_index]
                self.x_opt = new_harmony[min_index]

            self.convergence_curve.append(1.0 / (1.0 + self.f_opt))

        return self.f_opt, self.x_opt, self.convergence_curve

    def generate_new_harmony(self, harmony_memory, func, t):
        new_harmony = np.copy(harmony_memory)

        for i in range(len(func.bounds.lb)):
            index = np.random.choice(self.harmony_memory_size, size=2, replace=False)
            new_value = (harmony_memory[index[0], i] + harmony_memory[index[1], i]) / 2
            new_value = np.clip(new_value, func.bounds.lb[i], func.bounds.ub[i])
            new_harmony[:, i] = new_value

            if np.random.rand() < self.global_best_rate:
                global_best_index = np.argmin([func(x) for x in harmony_memory])
                new_harmony[:, i] = harmony_memory[global_best_index, i]

            levy = self.generate_levy_flight(len(func.bounds.lb), self.levy_step_size)
            new_harmony[:, i] += levy[:, i]

        return new_harmony

    def generate_levy_flight(self, dimension, step_size):
        levy = np.zeros((self.harmony_memory_size, dimension))
        epsilon = 1e-6
        sigma = (
            np.math.gamma(1.5) * np.sin(np.pi * 1.5 / 2) / (np.math.gamma(1.75) * 1.5 * 2**0.25)
        ) ** 0.6667

        for i in range(dimension):
            u = np.random.normal(0, sigma, self.harmony_memory_size)
            v = np.random.normal(0, 1, self.harmony_memory_size)
            step = u / (np.abs(v) ** 0.6667 + epsilon)
            levy[:, i] = 1.5 * step * step_size

        return levy
