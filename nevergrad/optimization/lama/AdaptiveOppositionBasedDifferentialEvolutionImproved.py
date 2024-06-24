import numpy as np


class AdaptiveOppositionBasedDifferentialEvolutionImproved:
    def __init__(self, budget=10000, pop_size=20, f_min=0.4, f_max=0.9, cr_min=0.1, cr_max=0.9):
        self.budget = budget
        self.pop_size = pop_size
        self.f_min = f_min
        self.f_max = f_max
        self.cr_min = cr_min
        self.cr_max = cr_max
        self.dim = 5
        self.f_opt = np.Inf
        self.x_opt = None

    def initialize_population(self, func):
        self.population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        self.pop_fitness = np.array([func(x) for x in self.population])

    def opposition_based_learning(self, solution, bounds):
        return 2 * bounds.lb - solution + 2 * (solution - bounds.lb)

    def differential_evolution(self, func, current_solution, best_solution, f, cr):
        mutant_solution = current_solution + f * (best_solution - current_solution)
        crossover_mask = np.random.rand(self.dim) < cr
        trial_solution = np.where(crossover_mask, mutant_solution, current_solution)
        return np.clip(trial_solution, func.bounds.lb, func.bounds.ub)

    def self_adaptive_parameter_update(self, success, f, cr):
        f_new = f * (1.0 + 0.1 * (1.0 if success else -1.0))
        cr_new = cr + 0.1 * (0.9 if success else -0.9)
        return max(self.f_min, min(self.f_max, f_new)), max(self.cr_min, min(self.cr_max, cr_new))

    def __call__(self, func):
        self.initialize_population(func)
        f_current = (self.f_min + self.f_max) / 2
        cr_current = (self.cr_min + self.cr_max) / 2

        for _ in range(self.budget):
            idx = np.argsort(self.pop_fitness)
            best_solution = self.population[idx[0]]

            for j in range(self.pop_size):
                current_solution = self.population[j]

                opponent_solution = self.opposition_based_learning(current_solution, func.bounds)
                f_current, cr_current = self.self_adaptive_parameter_update(True, f_current, cr_current)

                trial_solution = self.differential_evolution(
                    func, current_solution, best_solution, f_current, cr_current
                )

                if func(trial_solution) < func(current_solution):
                    self.population[j] = trial_solution
                    self.pop_fitness[j] = func(trial_solution)
                    f_current, cr_current = self.self_adaptive_parameter_update(True, f_current, cr_current)
                else:
                    f_current, cr_current = self.self_adaptive_parameter_update(False, f_current, cr_current)

                if func(opponent_solution) < func(current_solution):
                    self.population[j] = opponent_solution
                    self.pop_fitness[j] = func(opponent_solution)
                    f_current, cr_current = self.self_adaptive_parameter_update(True, f_current, cr_current)
                else:
                    f_current, cr_current = self.self_adaptive_parameter_update(False, f_current, cr_current)

                if func(trial_solution) < self.f_opt:
                    self.f_opt = func(trial_solution)
                    self.x_opt = trial_solution

        return self.f_opt, self.x_opt
