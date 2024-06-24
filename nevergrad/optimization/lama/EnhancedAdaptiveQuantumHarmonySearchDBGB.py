import numpy as np
from scipy.stats import cauchy


class EnhancedAdaptiveQuantumHarmonySearchDBGB:
    def __init__(self, budget=1000, hmcr=0.7, par=0.3, init_bw=0.1, bw_range=[0.01, 0.2], bw_decay=0.95):
        self.budget = budget
        self.hmcr = hmcr
        self.par = par
        self.init_bw = init_bw
        self.bw_range = bw_range
        self.bw_decay = bw_decay

    def cauchy_mutation(self, value, lb, ub, scale=0.1):
        mutated_value = value + cauchy.rvs(loc=0, scale=scale)
        mutated_value = np.clip(mutated_value, lb, ub)
        return mutated_value

    def adaptive_bandwidth(self, iteration):
        return max(self.init_bw * (self.bw_decay**iteration), self.bw_range[0])

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        harmony_memory = np.random.uniform(
            func.bounds.lb, func.bounds.ub, size=(self.budget, len(func.bounds.lb))
        )
        global_best = harmony_memory[np.argmin([func(h) for h in harmony_memory])]
        bandwidth = self.init_bw

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
                        new_harmony[j], func.bounds.lb[j], func.bounds.ub[j], scale=bandwidth
                    )

            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony

            if f < func(global_best):
                global_best = new_harmony

            if np.random.rand() < 0.1:
                new_harmony = np.random.uniform(func.bounds.lb, func.bounds.ub)
                f = func(new_harmony)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = new_harmony

            bandwidth = self.adaptive_bandwidth(i)

        return self.f_opt, self.x_opt
