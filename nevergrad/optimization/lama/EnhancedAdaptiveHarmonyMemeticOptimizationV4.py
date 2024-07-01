import numpy as np


class EnhancedAdaptiveHarmonyMemeticOptimizationV4:
    def __init__(
        self,
        budget=10000,
        memetic_iter=100,
        memetic_prob=0.9,
        memetic_step=0.1,
        memory_size=50,
        pitch_adjustment_rate=0.5,
    ):
        self.budget = budget
        self.dim = 5
        self.memetic_iter = memetic_iter
        self.memetic_prob = memetic_prob
        self.memetic_step = memetic_step
        self.memory_size = memory_size
        self.pitch_adjustment_rate = pitch_adjustment_rate

    def _initialize_harmony_memory(self, func):
        harmony_memory = [np.random.uniform(-5.0, 5.0, size=self.dim) for _ in range(self.memory_size)]
        harmony_memory_costs = [func(hm) for hm in harmony_memory]
        return harmony_memory, harmony_memory_costs

    def _improvise_new_harmony(self, harmony_memory):
        new_harmony = np.empty(self.dim)
        for i in range(self.dim):
            new_harmony[i] = np.random.choice([hm[i] for hm in harmony_memory])
            if np.random.rand() < self.pitch_adjustment_rate:
                new_harmony[i] += np.random.normal(0, 0.5)
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

        return best_harmony, best_cost

    def _apply_memetic_search(self, harmony_memory, harmony_memory_costs, func):
        for idx in range(len(harmony_memory)):
            if np.random.rand() < self.memetic_prob:
                harmony_memory[idx], harmony_memory_costs[idx] = self._memetic_local_search(
                    harmony_memory[idx], func
                )

        return harmony_memory, harmony_memory_costs

    def _harmony_selection(self, harmony_memory, harmony_memory_costs):
        idx = np.argsort(harmony_memory_costs)[: self.memory_size]
        return [harmony_memory[i] for i in idx], [harmony_memory_costs[i] for i in idx]

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        harmony_memory, harmony_memory_costs = self._initialize_harmony_memory(func)

        for i in range(self.budget):
            new_harmony = self._improvise_new_harmony(harmony_memory)

            if np.random.rand() < 0.1:
                new_harmony = np.random.uniform(-5.0, 5.0, size=self.dim)

            if np.random.rand() < 0.8:
                new_harmony, new_cost = self._memetic_local_search(new_harmony, func)
            else:
                new_cost = func(new_harmony)

            harmony_memory.append(new_harmony)
            harmony_memory_costs.append(new_cost)

            harmony_memory, harmony_memory_costs = self._apply_memetic_search(
                harmony_memory, harmony_memory_costs, func
            )

            harmony_memory, harmony_memory_costs = self._harmony_selection(
                harmony_memory, harmony_memory_costs
            )

            if new_cost < self.f_opt:
                self.f_opt = new_cost
                self.x_opt = new_harmony

        return 1.0 - np.mean(np.array(harmony_memory_costs)), np.std(np.array(harmony_memory_costs))
