import numpy as np


class ImprovedOppositionBasedDifferentialEvolution:
    def __init__(self, budget=10000, pop_size=20, f_init=0.5, cr_init=0.9, scaling_factor=0.1, p_best=0.25):
        self.budget = budget
        self.pop_size = pop_size
        self.f_init = f_init
        self.cr_init = cr_init
        self.scaling_factor = scaling_factor
        self.p_best = p_best
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
        f_scale = scaling_factor * (1.0 - 2.0 * np.random.rand())
        cr_scale = scaling_factor * (1.0 - 2.0 * np.random.rand())
        f_new = np.clip(f + f_scale, 0.0, 1.0)
        cr_new = np.clip(cr + cr_scale, 0.0, 1.0)

        return f_new, cr_new

    def update_best_solution(self, current_fitness, trial_fitness, current_solution, trial_solution):
        if trial_fitness < current_fitness:
            return trial_solution, trial_fitness
        else:
            return current_solution, current_fitness

    def __call__(self, func):
        self.initialize_population(func)
        f_current = self.f_init
        cr_current = self.cr_init

        for _ in range(self.budget):
            idx = np.argsort(self.pop_fitness)
            best_solution = self.population[idx[0]]

            for j in range(self.pop_size):
                current_solution = self.population[j]

                opponent_solution = self.opposition_based_learning(current_solution, func.bounds)
                trial_solution = self.differential_evolution(
                    func, current_solution, best_solution, f_current, cr_current
                )

                trial_fitness = func(trial_solution)
                opponent_fitness = func(opponent_solution)

                if trial_fitness < self.pop_fitness[j]:
                    self.population[j] = trial_solution
                    self.pop_fitness[j] = trial_fitness

                if opponent_fitness < self.pop_fitness[j]:
                    self.population[j] = opponent_solution
                    self.pop_fitness[j] = opponent_fitness

                f_current, cr_current = self.adaptive_parameter_update(
                    trial_fitness < self.pop_fitness[j] or opponent_fitness < self.pop_fitness[j],
                    f_current,
                    cr_current,
                    self.scaling_factor,
                )

                self.population[j], self.pop_fitness[j] = self.update_best_solution(
                    self.pop_fitness[j], trial_fitness, self.population[j], trial_solution
                )
                self.population[j], self.pop_fitness[j] = self.update_best_solution(
                    self.pop_fitness[j], opponent_fitness, self.population[j], opponent_solution
                )

                if self.pop_fitness[j] < self.f_opt:
                    self.f_opt = self.pop_fitness[j]
                    self.x_opt = self.population[j]

        return self.f_opt, self.x_opt
