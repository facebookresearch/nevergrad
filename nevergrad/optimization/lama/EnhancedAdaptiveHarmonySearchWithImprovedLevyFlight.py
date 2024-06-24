import numpy as np


class EnhancedAdaptiveHarmonySearchWithImprovedLevyFlight:
    def __init__(self, budget, harmony_memory_size=20, global_best_rate=0.1, leviness=1.5, alpha=0.6):
        self.budget = budget
        self.harmony_memory_size = harmony_memory_size
        self.global_best_rate = global_best_rate
        self.leviness = leviness
        self.alpha = alpha
        self.convergence_curve = []

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        harmony_memory = np.random.uniform(
            func.bounds.lb, func.bounds.ub, size=(self.harmony_memory_size, len(func.bounds.lb))
        )

        for t in range(self.budget):
            new_harmony = self.generate_new_harmony(harmony_memory, func)
            new_harmony_fitness = np.array([func(x) for x in new_harmony])

            min_index = np.argmin(new_harmony_fitness)
            if new_harmony_fitness[min_index] < self.f_opt:
                self.f_opt = new_harmony_fitness[min_index]
                self.x_opt = new_harmony[min_index]

            self.convergence_curve.append(1.0 / (1.0 + self.f_opt))

        return self.f_opt, self.x_opt, self.convergence_curve

    def generate_new_harmony(self, harmony_memory, func):
        new_harmony = np.copy(harmony_memory)

        for i in range(len(func.bounds.lb)):
            index = np.random.choice(self.harmony_memory_size, size=2, replace=False)
            new_value = (harmony_memory[index[0], i] + harmony_memory[index[1], i]) / 2
            new_value = np.clip(new_value, func.bounds.lb[i], func.bounds.ub[i])
            new_harmony[:, i] = new_value

            if np.random.rand() < self.global_best_rate:
                global_best_index = np.argmin([func(x) for x in harmony_memory])
                new_harmony[:, i] = harmony_memory[global_best_index, i]

            levy = self.generate_levy_flight(size=self.harmony_memory_size, dimension=len(func.bounds.lb))
            new_harmony[:, i] += self.alpha * levy[:, i]

        new_harmony = np.clip(new_harmony, func.bounds.lb, func.bounds.ub)

        return new_harmony

    def generate_levy_flight(self, size, dimension):
        levy = np.zeros((size, dimension))
        epsilon = 1e-6
        sigma = (
            np.math.gamma(1.0 + self.leviness)
            * np.sin(np.pi * self.leviness / 2)
            / (np.math.gamma(1.0 + 2 * self.leviness) * (self.leviness**0.5))
        ) ** (1.0 / self.leviness)

        for i in range(size):
            for j in range(dimension):
                u = np.random.normal(0, sigma)
                v = np.random.normal(0, 1)
                step = u / (np.abs(v) ** (1.0 / self.leviness) + epsilon)
                levy[i, j] = step

        return levy
