import numpy as np


class AdaptiveEnhancedHarmonySearchWithLevyFlightInspiration:
    def __init__(self, budget, harmony_memory_size=10, bandwidth=0.1, mutation_rate=0.2):
        self.budget = budget
        self.harmony_memory_size = harmony_memory_size
        self.bandwidth = bandwidth
        self.mutation_rate = mutation_rate

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        harmony_memory = np.random.uniform(
            func.bounds.lb, func.bounds.ub, size=(self.harmony_memory_size, len(func.bounds.lb))
        )
        convergence_curve = []

        for _ in range(self.budget):
            new_harmony = self.generate_new_harmony(harmony_memory, func)
            new_harmony_fitness = np.array([func(x) for x in new_harmony])

            min_index = np.argmin(new_harmony_fitness)
            if new_harmony_fitness[min_index] < self.f_opt:
                self.f_opt = new_harmony_fitness[min_index]
                self.x_opt = new_harmony[min_index]

            convergence_curve.append(self.f_opt)

        return self.f_opt, self.x_opt, convergence_curve

    def generate_new_harmony(self, harmony_memory, func):
        new_harmony = np.copy(harmony_memory)
        for i in range(len(func.bounds.lb)):
            if np.random.rand() < 0.5:
                index = np.random.choice(self.harmony_memory_size, size=2, replace=False)
                new_value = np.clip(
                    np.random.normal(
                        (harmony_memory[index[0], i] + harmony_memory[index[1], i]) / 2, self.bandwidth
                    ),
                    func.bounds.lb[i],
                    func.bounds.ub[i],
                )
                new_harmony[:, i] = new_value

            if np.random.rand() < self.mutation_rate:
                mutation_indices = np.random.choice(
                    self.harmony_memory_size,
                    size=int(self.mutation_rate * self.harmony_memory_size),
                    replace=False,
                )
                for idx in mutation_indices:
                    new_harmony[idx, i] = np.random.uniform(func.bounds.lb[i], func.bounds.ub[i])

            if np.random.rand() < 0.1:  # Introduce Adaptive Levy Flight
                levy = self.generate_adaptive_levy_flight(len(func.bounds.lb))
                new_harmony[:, i] += levy

        return new_harmony

    def generate_adaptive_levy_flight(self, dimension):
        alpha = 1.5  # tuning parameter
        beta = 0.5  # tuning parameter

        sigma = (
            np.math.gamma(1 + beta)
            * np.sin(np.pi * beta / 2)
            / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = np.random.normal(0, sigma, self.harmony_memory_size)
        v = np.random.normal(0, 1, self.harmony_memory_size)
        step = u / abs(v) ** (1 / beta)
        levy = step * alpha

        return levy
