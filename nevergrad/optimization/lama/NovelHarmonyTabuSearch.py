import numpy as np


class NovelHarmonyTabuSearch:
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

    def generate_new_solution(self, harmony_memory, best_solution, bounds, tabu_list):
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
            return self.generate_new_solution(harmony_memory, best_solution, bounds, tabu_list)

        return new_solution, new_solution_str

    def update_tabu_list(self, tabu_list, new_solution_str):
        tabu_list.append(new_solution_str)
        if len(tabu_list) > self.tabu_tenure:
            tabu_list.pop(0)

    def update_pitch_adjustment_rate(self, iteration):
        self.pitch_adjustment_rate = max(0.1, self.pitch_adjustment_rate - 0.1 * iteration / self.budget)

    def update_tabu_tenure(self, num_improvements):
        if num_improvements == 0.1 * self.budget:
            self.tabu_tenure += 1

    def evaluate_harmony(self, harmony, func):
        return func(harmony)

    def harmonize(self, func, harmony_memory):
        harmony_scores = [self.evaluate_harmony(harmony, func) for harmony in harmony_memory]
        best_index = np.argmin(harmony_scores)
        return harmony_memory[best_index], harmony_scores[best_index]

    def update_memory(self, harmony_memory, new_solution, func):
        worst_index = np.argmax([func(harmony) for harmony in harmony_memory])
        if func(new_solution) < func(harmony_memory[worst_index]):
            harmony_memory[worst_index] = new_solution

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        bounds = func.bounds

        harmony_memory = self.initialize_positions(bounds)
        tabu_list = []
        num_improvements = 0

        for i in range(self.budget):
            new_solution, new_solution_str = self.generate_new_solution(
                harmony_memory, self.x_opt, bounds, tabu_list
            )
            if new_solution_str not in tabu_list:
                self.update_memory(harmony_memory, new_solution, func)
                self.update_tabu_list(tabu_list, new_solution_str)
                self.update_pitch_adjustment_rate(i)

            best_harmony, best_score = self.harmonize(func, harmony_memory)
            if best_score < self.f_opt:
                self.f_opt = best_score
                self.x_opt = best_harmony
                num_improvements += 1

            self.update_tabu_tenure(num_improvements)

        return self.f_opt, self.x_opt
