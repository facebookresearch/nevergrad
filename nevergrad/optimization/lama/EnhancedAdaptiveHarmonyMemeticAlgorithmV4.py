import numpy as np


class EnhancedAdaptiveHarmonyMemeticAlgorithmV4:
    def __init__(
        self, budget=10000, hmcr=0.7, par=0.2, bw=0.05, memetic_iter=100, memetic_prob=0.8, memetic_step=0.03
    ):
        self.budget = budget
        self.dim = 5
        self.hmcr = hmcr
        self.par = par
        self.bw = bw
        self.memetic_iter = memetic_iter
        self.memetic_prob = memetic_prob
        self.memetic_step = memetic_step

    def _initialize_harmony_memory(self, func):
        harmony_memory = [np.random.uniform(-5.0, 5.0, size=self.dim) for _ in range(self.budget)]
        harmony_memory_costs = [func(hm) for hm in harmony_memory]
        return harmony_memory, harmony_memory_costs

    def _improvise_new_harmony(self, harmony_memory):
        new_harmony = np.empty(self.dim)
        for i in range(self.dim):
            if np.random.rand() < self.hmcr:
                if np.random.rand() < self.par:
                    new_harmony[i] = harmony_memory[np.random.randint(len(harmony_memory))][i]
                else:
                    new_harmony[i] = np.random.uniform(-5.0, 5.0)
            else:
                new_harmony[i] = np.random.uniform(-5.0, 5.0)

            if np.random.rand() < self.bw:
                new_harmony[i] += np.random.normal(0, 1)

            new_harmony[i] = np.clip(new_harmony[i], -5.0, 5.0)

        return new_harmony

    def _memetic_local_search(self, harmony, func):
        best_harmony = harmony.copy()
        best_cost = func(harmony)

        for _ in range(self.memetic_iter):
            mutated_harmony = harmony + np.random.normal(0, self.memetic_step, size=self.dim)
            mutated_harmony = np.clip(mutated_harmony, -5.0, 5.0)
            cost = func(mutated_harmony)

            if cost < best_cost:
                best_harmony = mutated_harmony
                best_cost = cost

        return best_harmony

    def _apply_memetic_search(self, harmony_memory, harmony_memory_costs, func):
        for idx in range(len(harmony_memory)):
            if np.random.rand() < self.memetic_prob:
                harmony_memory[idx] = self._memetic_local_search(harmony_memory[idx], func)
                harmony_memory_costs[idx] = func(harmony_memory[idx])

        return harmony_memory, harmony_memory_costs

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        harmony_memory, harmony_memory_costs = self._initialize_harmony_memory(func)

        for _ in range(self.budget):
            self.hmcr = max(self.hmcr - 0.0001, 0.6)  # Enhanced adjustment of HMCR
            self.par = min(self.par + 0.0001, 0.3)  # Enhanced adjustment of PAR
            self.bw = max(self.bw - 0.0001, 0.02)  # Enhanced adjustment of BW
            self.memetic_prob = min(
                self.memetic_prob + 0.0001, 1.0
            )  # Enhanced adjustment of Memetic Probability

            new_harmony = self._improvise_new_harmony(harmony_memory)
            new_harmony = self._memetic_local_search(new_harmony, func)
            new_cost = func(new_harmony)

            if new_cost < min(harmony_memory_costs):
                idx = np.argmin(harmony_memory_costs)
                harmony_memory[idx] = new_harmony
                harmony_memory_costs[idx] = new_cost

            harmony_memory, harmony_memory_costs = self._apply_memetic_search(
                harmony_memory, harmony_memory_costs, func
            )

            if new_cost < self.f_opt:
                self.f_opt = new_cost
                self.x_opt = new_harmony

        return self.f_opt, self.x_opt
