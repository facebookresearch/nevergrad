import numpy as np


class EnhancedProgressiveAdaptiveDifferentialEvolution:
    def __init__(self, budget, dim=5, pop_size=150, F_init=0.8, F_end=0.2, CR=0.8):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F_init = F_init  # Initial mutation factor
        self.F_end = F_end  # Final mutation factor during the later stages of optimization
        self.CR = CR  # Crossover probability
        self.bounds = (-5.0, 5.0)

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], size=(self.pop_size, self.dim))

    def mutate(self, population, idx, n_evals):
        indices = [i for i in range(self.pop_size) if i != idx]
        a, b, c = np.random.choice(indices, 3, replace=False)
        # Linear decrease of mutation factor from F_init to F_end
        F = self.F_init - (self.F_init - self.F_end) * (n_evals / self.budget)
        mutant = np.clip(population[a] + F * (population[b] - population[c]), self.bounds[0], self.bounds[1])
        return mutant

    def crossover(self, target, mutant):
        # Binomial crossover
        cross_points = np.random.rand(self.dim) < self.CR
        if not np.any(cross_points):
            cross_points[np.random.randint(0, self.dim)] = True
        trial = np.where(cross_points, mutant, target)
        return trial

    def select(self, population, f_values, trial, trial_f, trial_idx):
        if trial_f < f_values[trial_idx]:
            population[trial_idx] = trial
            f_values[trial_idx] = trial_f

    def __call__(self, func):
        population = self.initialize_population()
        f_values = np.array([func(ind) for ind in population])
        n_evals = self.pop_size

        while n_evals < self.budget:
            for idx in range(self.pop_size):
                mutant = self.mutate(population, idx, n_evals)
                trial = self.crossover(population[idx], mutant)
                trial_f = func(trial)
                n_evals += 1
                self.select(population, f_values, trial, trial_f, idx)
                if n_evals >= self.budget:
                    break

        self.f_opt = np.min(f_values)
        self.x_opt = population[np.argmin(f_values)]

        return self.f_opt, self.x_opt
