import numpy as np
from scipy.stats import cauchy


class OptimizedQuantumHarmonySearch:
    def __init__(self, budget=1000, hmcr=0.7, par=0.3, bw=0.1):
        self.budget = budget
        self.hmcr = hmcr  # Harmony Memory Considering Rate
        self.par = par  # Pitch Adjustment Rate
        self.bw = bw  # Bandwidth

    def cauchy_mutation(self, value, lb, ub, scale=0.1):
        mutated_value = value + cauchy.rvs(loc=0, scale=scale)
        mutated_value = np.clip(mutated_value, lb, ub)
        return mutated_value

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        harmony_memory = np.random.uniform(
            func.bounds.lb, func.bounds.ub, size=(self.budget, len(func.bounds.lb))
        )

        for i in range(self.budget):
            new_harmony = np.zeros(len(func.bounds.lb))
            for j in range(len(func.bounds.lb)):
                if np.random.rand() < self.hmcr:
                    idx = np.random.randint(0, len(harmony_memory))
                    new_harmony[j] = harmony_memory[idx, j]
                else:
                    new_harmony[j] = np.random.uniform(func.bounds.lb[j], func.bounds.ub[j])

                if np.random.rand() < self.par:
                    new_harmony[j] = self.cauchy_mutation(
                        new_harmony[j], func.bounds.lb[j], func.bounds.ub[j], scale=self.bw
                    )

            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony

        return self.f_opt, self.x_opt
