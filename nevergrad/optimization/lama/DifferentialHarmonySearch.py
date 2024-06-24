import numpy as np


class DifferentialHarmonySearch:
    def __init__(self, budget=10000, harmony_memory_size=20, hmcr=0.9, par=0.4, bw=0.5):
        self.budget = budget
        self.harmony_memory_size = harmony_memory_size
        self.hmcr = hmcr  # Harmony Memory Consideration Rate
        self.par = par  # Pitch Adjustment Rate
        self.bw = bw  # Bandwidth

        self.dim = 5
        self.f_opt = np.Inf
        self.x_opt = None

    def initialize_harmony_memory(self, func):
        self.harmony_memory = np.random.uniform(
            func.bounds.lb, func.bounds.ub, (self.harmony_memory_size, self.dim)
        )
        self.harmony_memory_fitness = np.array([func(x) for x in self.harmony_memory])

    def __call__(self, func):
        self.initialize_harmony_memory(func)

        for _ in range(self.budget):
            new_harmony = np.zeros(self.dim)
            for j in range(self.dim):
                if np.random.rand() < self.hmcr:
                    idx = np.random.randint(self.harmony_memory_size)
                    new_harmony[j] = self.harmony_memory[idx, j]
                else:
                    new_harmony[j] = np.random.uniform(func.bounds.lb[j], func.bounds.ub[j])

                if np.random.rand() < self.par:
                    new_harmony[j] += self.bw * np.random.randn()

                new_harmony[j] = np.clip(new_harmony[j], func.bounds.lb[j], func.bounds.ub[j])

            new_fitness = func(new_harmony)
            if new_fitness < self.f_opt:
                self.f_opt = new_fitness
                self.x_opt = new_harmony

            idx_worst = np.argmax(self.harmony_memory_fitness)
            if new_fitness < self.harmony_memory_fitness[idx_worst]:
                self.harmony_memory[idx_worst] = new_harmony
                self.harmony_memory_fitness[idx_worst] = new_fitness

        return self.f_opt, self.x_opt
