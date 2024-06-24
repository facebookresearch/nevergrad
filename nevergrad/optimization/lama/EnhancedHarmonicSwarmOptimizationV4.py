import numpy as np


class EnhancedHarmonicSwarmOptimizationV4:
    def __init__(
        self,
        budget=1000,
        num_particles=20,
        num_dimensions=5,
        harmony_memory_rate=0.6,
        pitch_adjust_rate=0.5,
        local_search_prob=0.5,
        step_size=0.2,
    ):
        self.budget = budget
        self.num_particles = num_particles
        self.num_dimensions = num_dimensions
        self.harmony_memory_rate = harmony_memory_rate
        self.pitch_adjust_rate = pitch_adjust_rate
        self.local_search_prob = local_search_prob
        self.step_size = step_size

    def initialize_positions(self, bounds):
        return np.random.uniform(bounds.lb, bounds.ub, size=(self.num_particles, self.num_dimensions))

    def generate_new_solution(self, memory_matrix, pitch_matrix, bounds):
        new_solution = np.zeros_like(memory_matrix[0])
        for i in range(self.num_dimensions):
            if np.random.rand() < self.pitch_adjust_rate:
                new_solution[i] = np.random.uniform(bounds.lb[i], bounds.ub[i])
            else:
                index = np.random.randint(self.num_particles)
                new_solution[i] = memory_matrix[index, i]

        return new_solution

    def local_search(self, solution, func, bounds):
        new_solution = solution.copy()
        for i in range(self.num_dimensions):
            if np.random.rand() < self.local_search_prob:
                new_solution[i] = np.clip(
                    new_solution[i] + np.random.normal(0, self.step_size), bounds.lb[i], bounds.ub[i]
                )
        if func(new_solution) < func(solution):
            return new_solution
        return solution

    def update_memory_matrix(self, memory_matrix, new_solution, func):
        worst_index = np.argmax([func(solution) for solution in memory_matrix])
        if func(new_solution) < func(memory_matrix[worst_index]):
            memory_matrix[worst_index] = new_solution

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        bounds = func.bounds

        memory_matrix = self.initialize_positions(bounds)

        for i in range(self.budget):
            new_solution = self.generate_new_solution(memory_matrix, memory_matrix, bounds)
            new_solution = self.local_search(new_solution, func, bounds)
            self.update_memory_matrix(memory_matrix, new_solution, func)

            if func(new_solution) < self.f_opt:
                self.f_opt = func(new_solution)
                self.x_opt = new_solution

        return self.f_opt, self.x_opt
