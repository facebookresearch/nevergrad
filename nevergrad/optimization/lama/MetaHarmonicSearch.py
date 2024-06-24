import numpy as np


class MetaHarmonicSearch:
    def __init__(
        self, budget=1000, num_agents=20, num_dimensions=5, harmony_memory_rate=0.7, pitch_adjust_rate=0.5
    ):
        self.budget = budget
        self.num_agents = num_agents
        self.num_dimensions = num_dimensions
        self.harmony_memory_rate = harmony_memory_rate
        self.pitch_adjust_rate = pitch_adjust_rate

    def initialize_agents(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, size=(self.num_agents, self.num_dimensions))

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        bounds = func.bounds

        harmony_memory = self.initialize_agents(bounds)
        best_harmony = harmony_memory[0].copy()

        for _ in range(self.budget):
            for i in range(self.num_agents):
                new_harmony = np.zeros(self.num_dimensions)
                for d in range(self.num_dimensions):
                    if np.random.rand() < self.harmony_memory_rate:
                        new_harmony[d] = harmony_memory[np.random.randint(self.num_agents)][d]
                    else:
                        new_harmony[d] = np.random.uniform(bounds.lb[d], bounds.ub[d])

                    if np.random.rand() < self.pitch_adjust_rate:
                        new_harmony[d] += np.random.uniform(-1, 1) * (bounds.ub[d] - bounds.lb[d])
                        new_harmony[d] = np.clip(new_harmony[d], bounds.lb[d], bounds.ub[d])

                f_new = func(new_harmony)
                if f_new < func(harmony_memory[i]):
                    harmony_memory[i] = new_harmony.copy()
                    if f_new < func(best_harmony):
                        best_harmony = new_harmony.copy()

        self.f_opt = func(best_harmony)
        self.x_opt = best_harmony

        return self.f_opt, self.x_opt
