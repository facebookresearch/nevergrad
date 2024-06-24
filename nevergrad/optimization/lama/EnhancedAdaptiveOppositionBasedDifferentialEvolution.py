import numpy as np


class EnhancedAdaptiveOppositionBasedDifferentialEvolution:
    def __init__(self, budget=10000, pop_size=40, f_init=0.8, cr_init=0.9, scaling_factor=0.1):
        self.budget = budget
        self.pop_size = pop_size
        self.f_init = f_init
        self.cr_init = cr_init
        self.scaling_factor = scaling_factor
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

    def adaptive_parameter_update(self, success, f, cr, scaling_factor):
        success_rate = success / self.pop_size
        f_scale = scaling_factor * (1 - 2 * np.random.rand()) * (1 - success_rate)
        cr_scale = scaling_factor * (1 - 2 * np.random.rand()) * (1 - success_rate)
        f_new = np.clip(f + f_scale, 0.0, 1.0)
        cr_new = np.clip(cr + cr_scale, 0.0, 1.0)

        return f_new, cr_new

    def update_best_solution(self, current_fitness, trial_fitness, current_solution, trial_solution):
        if trial_fitness < current_fitness:
            return trial_solution, trial_fitness
        else:
            return current_solution, current_fitness

    def enhance_search(self, solution, best_solution, scaling_factor):
        return solution + scaling_factor * (best_solution - solution)

    def __call__(self, func):
        self.initialize_population(func)
        f_current = self.f_init
        cr_current = self.cr_init

        for _ in range(self.budget):
            idx = np.argsort(self.pop_fitness)
            best_solution = self.population[idx[0]]

            success_count = 0
            for j in range(self.pop_size):
                current_solution = self.population[j]

                opponent_solution = self.opposition_based_learning(current_solution, func.bounds)
                trial_solution = self.differential_evolution(
                    func, current_solution, best_solution, f_current, cr_current
                )
                enhanced_solution = self.enhance_search(current_solution, best_solution, self.scaling_factor)

                trial_fitness = func(trial_solution)
                opponent_fitness = func(opponent_solution)
                enhanced_fitness = func(enhanced_solution)

                if trial_fitness < self.pop_fitness[j]:
                    self.population[j] = trial_solution
                    self.pop_fitness[j] = trial_fitness
                    success_count += 1

                if opponent_fitness < self.pop_fitness[j]:
                    self.population[j] = opponent_solution
                    self.pop_fitness[j] = opponent_fitness
                    success_count += 1

                if enhanced_fitness < self.pop_fitness[j]:
                    self.population[j] = enhanced_solution
                    self.pop_fitness[j] = enhanced_fitness
                    success_count += 1

                f_current, cr_current = self.adaptive_parameter_update(
                    success_count, f_current, cr_current, self.scaling_factor
                )

                self.population[j], self.pop_fitness[j] = self.update_best_solution(
                    self.pop_fitness[j], trial_fitness, self.population[j], trial_solution
                )
                self.population[j], self.pop_fitness[j] = self.update_best_solution(
                    self.pop_fitness[j], opponent_fitness, self.population[j], opponent_solution
                )
                self.population[j], self.pop_fitness[j] = self.update_best_solution(
                    self.pop_fitness[j], enhanced_fitness, self.population[j], enhanced_solution
                )

                if self.pop_fitness[j] < self.f_opt:
                    self.f_opt = self.pop_fitness[j]
                    self.x_opt = self.population[j]

        return self.f_opt, self.x_opt
