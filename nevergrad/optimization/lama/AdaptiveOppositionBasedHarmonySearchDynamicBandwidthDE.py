import numpy as np


class AdaptiveOppositionBasedHarmonySearchDynamicBandwidthDE:
    def __init__(
        self,
        budget=10000,
        harmony_memory_size=20,
        hmcr=0.7,
        par=0.4,
        bw=0.5,
        bw_min=0.01,
        bw_decay=0.995,
        bw_range=0.5,
        de_scale=0.5,
    ):
        self.budget = budget
        self.harmony_memory_size = harmony_memory_size
        self.hmcr = hmcr  # Harmony Memory Consideration Rate
        self.par = par  # Pitch Adjustment Rate
        self.bw = bw  # Initial Bandwidth
        self.bw_min = bw_min  # Minimum Bandwidth
        self.bw_decay = bw_decay  # Bandwidth decay rate
        self.bw_range = bw_range  # Bandwidth range for dynamic adjustment
        self.de_scale = de_scale  # Scale factor for differential evolution

        self.dim = 5
        self.f_opt = np.Inf
        self.x_opt = None

    def initialize_harmony_memory(self, func):
        self.harmony_memory = np.random.uniform(
            func.bounds.lb, func.bounds.ub, (self.harmony_memory_size, self.dim)
        )
        self.harmony_memory_fitness = np.array([func(x) for x in self.harmony_memory])

    def harmony_search(self, func, bandwidth):
        new_harmony = np.zeros(self.dim)
        for j in range(self.dim):
            if np.random.rand() < self.hmcr:
                idx = np.random.randint(self.harmony_memory_size)
                new_harmony[j] = self.harmony_memory[idx, j]
            else:
                new_harmony[j] = np.random.uniform(func.bounds.lb[j], func.bounds.ub[j])

            if np.random.rand() < self.par:
                new_harmony[j] += bandwidth * np.random.randn()

            new_harmony[j] = np.clip(new_harmony[j], func.bounds.lb[j], func.bounds.ub[j])

        return new_harmony

    def opposition_based_learning(self, solution, bounds):
        return 2 * bounds.lb - solution + 2 * (solution - bounds.lb)

    def adjust_bandwidth(self, iteration):
        return max(self.bw_range / (1 + iteration), self.bw_min)  # Dynamic adjustment of bandwidth

    def differential_evolution(self, func, current_harmony, best_harmony):
        mutant_harmony = current_harmony + self.de_scale * (best_harmony - current_harmony)
        return np.clip(mutant_harmony, func.bounds.lb, func.bounds.ub)

    def __call__(self, func):
        self.initialize_harmony_memory(func)

        for i in range(self.budget):
            self.bw = self.adjust_bandwidth(i)  # Update bandwidth dynamically

            new_harmony = self.harmony_search(func, self.bw)
            new_fitness = func(new_harmony)

            if new_fitness < self.f_opt:
                self.f_opt = new_fitness
                self.x_opt = new_harmony

            idx_worst = np.argmax(self.harmony_memory_fitness)
            if new_fitness < self.harmony_memory_fitness[idx_worst]:
                self.harmony_memory[idx_worst] = new_harmony
                self.harmony_memory_fitness[idx_worst] = new_fitness

            improved_harmony = self.opposition_based_learning(new_harmony, func.bounds)
            improved_fitness = func(improved_harmony)

            if improved_fitness < self.f_opt:
                self.f_opt = improved_fitness
                self.x_opt = improved_harmony

                idx_worst_improved = np.argmax(self.harmony_memory_fitness)
                if improved_fitness < self.harmony_memory_fitness[idx_worst_improved]:
                    self.harmony_memory[idx_worst_improved] = improved_harmony
                    self.harmony_memory_fitness[idx_worst_improved] = improved_fitness

            best_harmony = self.harmony_memory[np.argmin(self.harmony_memory_fitness)]
            trial_harmony = self.differential_evolution(func, new_harmony, best_harmony)
            trial_fitness = func(trial_harmony)

            if trial_fitness < self.f_opt:
                self.f_opt = trial_fitness
                self.x_opt = trial_harmony

                idx_worst_trial = np.argmax(self.harmony_memory_fitness)
                if trial_fitness < self.harmony_memory_fitness[idx_worst_trial]:
                    self.harmony_memory[idx_worst_trial] = trial_harmony
                    self.harmony_memory_fitness[idx_worst_trial] = trial_fitness

            self.bw = self.bw * self.bw_decay  # Decay bandwidth at each iteration

        return self.f_opt, self.x_opt
