import numpy as np


class AdaptiveEnhancedQuantumHarmonySearch:
    def __init__(
        self, budget, harmony_memory_size=10, pitch_adjustment_rate=0.1, bandwidth=0.01, mutation_rate=0.1
    ):
        self.budget = budget
        self.harmony_memory_size = harmony_memory_size
        self.pitch_adjustment_rate = pitch_adjustment_rate
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
            if np.random.rand() < self.pitch_adjustment_rate:
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

        return new_harmony
