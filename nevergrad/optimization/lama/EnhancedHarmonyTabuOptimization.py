import numpy as np


class EnhancedHarmonyTabuOptimization:
    def __init__(
        self,
        budget=1000,
        num_harmonies=50,
        num_dimensions=5,
        bandwidth=0.1,
        tabu_tenure=5,
        pitch_adjustment_rate=0.5,
    ):
        self.budget = budget
        self.num_harmonies = num_harmonies
        self.num_dimensions = num_dimensions
        self.bandwidth = bandwidth
        self.tabu_tenure = tabu_tenure
        self.pitch_adjustment_rate = pitch_adjustment_rate

    def initialize_positions(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, size=(self.num_harmonies, self.num_dimensions))

    def generate_new_solution(self, harmony_memory, bounds, tabu_list):
        new_solution = np.zeros_like(harmony_memory[0])
        for i in range(self.num_dimensions):
            if np.random.rand() < self.pitch_adjustment_rate:
                indexes = np.random.choice(range(self.num_harmonies), size=2, replace=False)
                new_solution[i] = np.mean(harmony_memory[indexes, i])
            else:
                new_solution[i] = np.random.uniform(bounds.lb[i], bounds.ub[i])

        new_solution = np.clip(new_solution, bounds.lb, bounds.ub)
        new_solution_str = ",".join(map(str, new_solution))
        if new_solution_str in tabu_list:
            return self.generate_new_solution(harmony_memory, bounds, tabu_list)

        return new_solution, new_solution_str

    def update_tabu_list(self, tabu_list, new_solution_str):
        tabu_list.append(new_solution_str)
        if len(tabu_list) > self.tabu_tenure:
            tabu_list.pop(0)

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        bounds = func.bounds

        harmony_memory = self.initialize_positions(bounds)
        tabu_list = []

        for _ in range(self.budget):
            new_solution, new_solution_str = self.generate_new_solution(harmony_memory, bounds, tabu_list)
            if new_solution_str not in tabu_list:
                harmony_memory[np.argmax([func(h) for h in harmony_memory])] = new_solution
                self.update_tabu_list(tabu_list, new_solution_str)

            best_index = np.argmin([func(h) for h in harmony_memory])
            if func(harmony_memory[best_index]) < self.f_opt:
                self.f_opt = func(harmony_memory[best_index])
                self.x_opt = harmony_memory[best_index]

        return self.f_opt, self.x_opt
