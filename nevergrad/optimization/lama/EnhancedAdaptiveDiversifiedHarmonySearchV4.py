import numpy as np
from scipy.stats import cauchy


class EnhancedAdaptiveDiversifiedHarmonySearchV4:
    def __init__(self, budget=1000, hmcr=0.7, par=0.6, init_bw=0.1, bw_range=[0.01, 0.2], bw_decay=0.98):
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

    def explore(self, func_bounds):
        return np.random.uniform(func_bounds.lb, func_bounds.ub)

    def exploit(self, harmony_memory, func, func_bounds, bandwidth):
        new_harmony = np.zeros(len(func_bounds.lb))
        for j in range(len(func_bounds.lb)):
            if np.random.rand() < self.hmcr:
                idx = np.random.randint(0, len(harmony_memory))
                new_harmony[j] = harmony_memory[idx][j]
            else:
                new_harmony[j] = np.random.uniform(func_bounds.lb[j], func_bounds.ub[j])

            if np.random.rand() < self.par:
                new_harmony[j] = self.cauchy_mutation(
                    new_harmony[j], func_bounds.lb[j], func_bounds.ub[j], scale=bandwidth
                )

        return new_harmony

    def global_best_update(self, harmony_memory, func):
        return harmony_memory[np.argmin([func(h) for h in harmony_memory])]

    def diversify_population(self, harmony_memory, func, func_bounds):
        for i in range(len(harmony_memory)):
            if np.random.rand() < 0.2:
                new_harmony = np.random.uniform(func_bounds.lb, func_bounds.ub)
                if func(new_harmony) < func(harmony_memory[i]):
                    harmony_memory[i] = new_harmony

    def local_optimization(self, solution, func, func_bounds, max_iter=10):
        current_solution = solution.copy()
        for _ in range(max_iter):
            new_solution = self.exploit(
                [current_solution], func, func.bounds, bandwidth=0.05
            )  # Fixed bandwidth for local search
            if func(new_solution) < func(current_solution):
                current_solution = new_solution
        return current_solution

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        harmony_memory = [np.random.uniform(func.bounds.lb, func.bounds.ub) for _ in range(self.budget)]
        global_best = self.global_best_update(harmony_memory, func)
        bandwidth = self.init_bw

        for i in range(self.budget):
            new_harmony = self.exploit(harmony_memory, func, func.bounds, bandwidth)
            new_harmony = self.local_optimization(new_harmony, func, func.bounds)
            f = func(new_harmony)
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = new_harmony

            global_best = self.global_best_update(harmony_memory, func)

            if np.random.rand() < 0.7:
                global_best = self.global_best_update(harmony_memory, func)

            bandwidth = self.adaptive_bandwidth(i)

            if i % 20 == 0:
                new_harmony = self.explore(func.bounds)
                f = func(new_harmony)
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = new_harmony

            if i % 50 == 0:
                self.diversify_population(harmony_memory, func, func.bounds)

        return self.f_opt, self.x_opt
