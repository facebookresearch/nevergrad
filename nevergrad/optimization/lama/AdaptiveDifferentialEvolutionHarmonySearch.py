import numpy as np


class AdaptiveDifferentialEvolutionHarmonySearch:
    def __init__(
        self,
        budget=10000,
        harmony_memory_size=20,
        hmcr=0.9,
        par=0.4,
        bw=0.5,
        bw_decay=0.95,
        f_weight=0.8,
        cr=0.9,
    ):
        self.budget = budget
        self.harmony_memory_size = harmony_memory_size
        self.hmcr = hmcr  # Harmony Memory Consideration Rate
        self.par = par  # Pitch Adjustment Rate
        self.bw = bw  # Bandwidth
        self.bw_decay = bw_decay  # Bandwidth decay rate
        self.f_weight = f_weight  # Differential evolution weighting factor
        self.cr = cr  # Differential evolution crossover rate

        self.dim = 5
        self.f_opt = np.Inf
        self.x_opt = None

    def initialize_harmony_memory(self, func):
        self.harmony_memory = np.random.uniform(
            func.bounds.lb, func.bounds.ub, (self.harmony_memory_size, self.dim)
        )
        self.harmony_memory_fitness = np.array([func(x) for x in self.harmony_memory])

    def differential_evolution(self, func):
        idxs = np.random.choice(self.harmony_memory_size, 3, replace=False)
        v = self.harmony_memory[idxs[0]] + self.f_weight * (
            self.harmony_memory[idxs[1]] - self.harmony_memory[idxs[2]]
        )
        mask = np.random.rand(self.dim) < self.cr
        if not np.any(mask):
            mask[np.random.randint(0, self.dim)] = True
        u = np.where(mask, v, self.harmony_memory[np.random.randint(0, self.harmony_memory_size)])

        return u

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

            de_harmony = self.differential_evolution(func)
            de_fitness = func(de_harmony)
            if de_fitness < self.f_opt:
                self.f_opt = de_fitness
                self.x_opt = de_harmony

            idx_worst_de = np.argmax(self.harmony_memory_fitness)
            if de_fitness < self.harmony_memory_fitness[idx_worst_de]:
                self.harmony_memory[idx_worst_de] = de_harmony
                self.harmony_memory_fitness[idx_worst_de] = de_fitness

            self.bw *= self.bw_decay  # Decay the bandwidth

        return self.f_opt, self.x_opt
