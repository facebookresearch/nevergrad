import numpy as np


class EnhancedAdaptiveHybridHarmonySearchV24:
    def __init__(
        self,
        budget,
        harmony_memory_size=20,
        levy_alpha=1.5,
        levy_beta=1.5,
        gaussian_std=0.1,
        levy_rate=0.3,
        levy_step_size=0.5,
    ):
        self.budget = budget
        self.harmony_memory_size = harmony_memory_size
        self.levy_alpha = levy_alpha
        self.levy_beta = levy_beta
        self.gaussian_std = gaussian_std
        self.levy_rate = levy_rate
        self.levy_step_size = levy_step_size
        self.convergence_curve = []

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        harmony_memory = np.random.uniform(
            func.bounds.lb, func.bounds.ub, size=(self.harmony_memory_size, len(func.bounds.lb))
        )

        for _ in range(self.budget):
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
        mutation_rate = self.compute_mutation_rate()

        for i in range(len(func.bounds.lb)):
            if np.random.rand() < 0.5:
                index = np.random.choice(self.harmony_memory_size, size=2, replace=False)
                new_value = np.clip(
                    np.random.normal(
                        (harmony_memory[index[0], i] + harmony_memory[index[1], i]) / 2, mutation_rate
                    ),
                    func.bounds.lb[i],
                    func.bounds.ub[i],
                )
                new_harmony[:, i] = new_value

            if np.random.rand() < self.levy_rate:
                levy = self.generate_levy_flight(len(func.bounds.lb))
                new_harmony[:, i] += levy[:, i]

            if np.random.rand() < 0.5:
                new_harmony[:, i] = np.clip(
                    new_harmony[:, i] + np.random.normal(0, self.gaussian_std, self.harmony_memory_size),
                    func.bounds.lb[i],
                    func.bounds.ub[i],
                )

        return new_harmony

    def generate_levy_flight(self, dimension):
        levy = np.zeros((self.harmony_memory_size, dimension))
        epsilon = 1e-6
        sigma = (
            np.math.gamma(1 + self.levy_beta)
            * np.sin(np.pi * self.levy_beta / 2)
            / (np.math.gamma((1 + self.levy_beta) / 2) * self.levy_beta * 2 ** ((self.levy_beta - 1) / 2))
        ) ** (1 / self.levy_beta)

        for i in range(dimension):
            u = np.random.normal(0, sigma, self.harmony_memory_size)
            v = np.random.normal(0, 1, self.harmony_memory_size)
            step = u / (np.abs(v) ** (1 / self.levy_beta + epsilon))
            levy[:, i] = self.levy_alpha * step * self.levy_step_size

        return levy

    def compute_mutation_rate(self):
        return np.exp(-1.0 * len(self.convergence_curve) / self.budget)
