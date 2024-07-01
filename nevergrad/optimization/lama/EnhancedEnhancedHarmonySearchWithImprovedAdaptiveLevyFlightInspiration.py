import numpy as np


class EnhancedEnhancedHarmonySearchWithImprovedAdaptiveLevyFlightInspiration:
    def __init__(
        self,
        budget,
        harmony_memory_size=15,
        bandwidth_min=0.1,
        bandwidth_max=1.0,
        mutation_rate=0.2,
        levy_iterations=5,
        levy_alpha=1.0,
        levy_beta_min=1.0,
        levy_beta_max=2.0,
    ):
        self.budget = budget
        self.harmony_memory_size = harmony_memory_size
        self.bandwidth_min = bandwidth_min
        self.bandwidth_max = bandwidth_max
        self.mutation_rate = mutation_rate
        self.levy_iterations = levy_iterations
        self.levy_alpha = levy_alpha
        self.levy_beta_min = levy_beta_min
        self.levy_beta_max = levy_beta_max

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

            convergence_curve.append(1.0 / (1.0 + self.f_opt))  # Calculate AOCC

        return self.f_opt, self.x_opt, convergence_curve

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

            if np.random.rand() < 0.1:  # Introduce Improved Adaptive Levy Flight
                levy = self.generate_improved_adaptive_levy_flight(len(func.bounds.lb))
                new_harmony[:, i] += levy

        return new_harmony

    def generate_improved_adaptive_levy_flight(self, dimension):
        beta = np.random.uniform(self.levy_beta_min, self.levy_beta_max)  # Randomly select beta in range
        sigma = (
            np.math.gamma(1 + beta)
            * np.sin(np.pi * beta / 2)
            / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)

        levy = np.zeros(self.harmony_memory_size)
        for _ in range(self.levy_iterations):
            u = np.random.normal(0, sigma, self.harmony_memory_size)
            v = np.random.normal(0, 1, self.harmony_memory_size)
            step = u / abs(v) ** (1 / beta)
            levy += step * self.levy_alpha
            beta *= 1.1  # Adjust beta for next iteration with smaller increment
            sigma = (
                np.math.gamma(1 + beta)
                * np.sin(np.pi * beta / 2)
                / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
            ) ** (1 / beta)

        return levy
