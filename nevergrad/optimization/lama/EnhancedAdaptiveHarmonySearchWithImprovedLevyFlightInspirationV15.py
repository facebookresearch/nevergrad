import numpy as np


class EnhancedAdaptiveHarmonySearchWithImprovedLevyFlightInspirationV15:
    def __init__(
        self,
        budget,
        harmony_memory_size=20,
        bandwidth_min=0.1,
        bandwidth_max=1.0,
        mutation_rate=0.2,
        levy_alpha=1.5,
        levy_beta=1.5,
    ):
        self.budget = budget
        self.harmony_memory_size = harmony_memory_size
        self.bandwidth_min = bandwidth_min
        self.bandwidth_max = bandwidth_max
        self.mutation_rate = mutation_rate
        self.levy_alpha = levy_alpha
        self.levy_beta = levy_beta
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

            self.convergence_curve.append(1.0 / (1.0 + self.f_opt))  # Calculate AOCC

        return self.f_opt, self.x_opt, self.convergence_curve

    def generate_new_harmony(self, harmony_memory, func):
        new_harmony = np.copy(harmony_memory)
        bandwidth = self.bandwidth_min + (self.bandwidth_max - self.bandwidth_min) * np.random.rand()

        for i in range(len(func.bounds.lb)):
            if np.random.rand() < 0.5:
                index = np.random.choice(self.harmony_memory_size, size=2, replace=False)
                new_value = np.clip(
                    np.random.normal(
                        (harmony_memory[index[0], i] + harmony_memory[index[1], i]) / 2, bandwidth
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

            if np.random.rand() < 0.1:
                levy = self.generate_improved_levy_flight(len(func.bounds.lb))
                new_harmony[:, i] += levy[:, i]

        return new_harmony

    def generate_improved_levy_flight(self, dimension):
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
            levy[:, i] = self.levy_alpha * step

        return levy
