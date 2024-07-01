import numpy as np


class SimulatedAnnealingOptimizer:
    def __init__(self, budget, initial_temperature=100.0, cooling_rate=0.99):
        self.budget = budget
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate

    def acceptance_probability(self, cost, new_cost, temperature):
        if new_cost < cost:
            return 1.0
        return np.exp((cost - new_cost) / temperature)

    def perturb_solution(self, current_solution, func, temperature):
        candidate_solution = current_solution + np.random.normal(0, 0.1, size=current_solution.shape)
        candidate_solution = np.clip(candidate_solution, func.bounds.lb, func.bounds.ub)
        return candidate_solution

    def __call__(self, func):
        temperature = self.initial_temperature
        current_solution = np.random.uniform(func.bounds.lb, func.bounds.ub)
        current_cost = func(current_solution)
        best_solution = current_solution
        best_cost = current_cost

        for _ in range(self.budget):
            new_solution = self.perturb_solution(current_solution, func, temperature)
            new_cost = func(new_solution)

            if self.acceptance_probability(current_cost, new_cost, temperature) > np.random.rand():
                current_solution = new_solution
                current_cost = new_cost

            if new_cost < best_cost:
                best_solution = new_solution
                best_cost = new_cost

            temperature *= self.cooling_rate

        return best_cost, best_solution
