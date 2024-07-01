import numpy as np


class SelfAdaptiveOppositionBasedHarmonySearchDE:
    def __init__(
        self, budget=10000, harmony_memory_size=20, hmcr=0.7, par=0.4, de_sf=0.8, de_cr=0.5, de_step_size=0.1
    ):
        self.budget = budget
        self.harmony_memory_size = harmony_memory_size
        self.hmcr = hmcr
        self.par = par
        self.de_sf = de_sf
        self.de_cr = de_cr
        self.de_step_size = de_step_size

        self.dim = 5
        self.f_opt = np.Inf
        self.x_opt = None

    def initialize_harmony_memory(self, func):
        self.harmony_memory = np.random.uniform(
            func.bounds.lb, func.bounds.ub, (self.harmony_memory_size, self.dim)
        )
        self.harmony_memory_fitness = np.array([func(x) for x in self.harmony_memory])

    def harmony_search(self, func):
        new_harmony = np.zeros(self.dim)
        for j in range(self.dim):
            if np.random.rand() < self.hmcr:
                idx = np.random.randint(self.harmony_memory_size)
                new_harmony[j] = self.harmony_memory[idx, j]
            else:
                new_harmony[j] = np.random.uniform(func.bounds.lb[j], func.bounds.ub[j])

            if np.random.rand() < self.par:
                new_harmony[j] += np.random.uniform(-1, 1) * self.de_step_size

            new_harmony[j] = np.clip(new_harmony[j], func.bounds.lb[j], func.bounds.ub[j])

        return new_harmony

    def opposition_based_learning(self, solution, bounds):
        return 2 * bounds.lb - solution + 2 * (solution - bounds.lb)

    def differential_evolution(self, func, current_harmony, best_harmony):
        mutant_harmony = current_harmony + self.de_sf * (best_harmony - current_harmony)
        crossover_mask = np.random.rand(self.dim) < self.de_cr
        trial_harmony = np.where(crossover_mask, mutant_harmony, current_harmony)
        return np.clip(trial_harmony, func.bounds.lb, func.bounds.ub)

    def self_adaptive_parameter_update(self, success):
        if success:
            self.hmcr = min(1.0, self.hmcr * 1.05)  # Increase HMCR if successful
            self.par = max(0.0, self.par * 0.95)  # Decrease PAR if successful
            self.de_sf = max(0.5, self.de_sf * 1.05)  # Increase DE scale factor if successful
            self.de_cr = min(1.0, self.de_cr * 1.05)  # Increase DE crossover rate if successful
        else:
            self.hmcr = max(0.0, self.hmcr * 0.95)  # Decrease HMCR if not successful
            self.par = min(1.0, self.par * 1.05)  # Increase PAR if not successful
            self.de_sf = max(0.5, self.de_sf * 0.95)  # Decrease DE scale factor if not successful
            self.de_cr = max(0.0, self.de_cr * 0.95)  # Decrease DE crossover rate if not successful

    def __call__(self, func):
        self.initialize_harmony_memory(func)

        for i in range(self.budget):
            new_harmony = self.harmony_search(func)
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
                    self.self_adaptive_parameter_update(True)
            else:
                self.self_adaptive_parameter_update(False)

        return self.f_opt, self.x_opt
