import numpy as np


class DifferentialEvolutionAdaptiveCrossover:
    def __init__(self, budget=10000, population_size=50, F=0.8, CR=0.9):
        self.budget = budget
        self.population_size = population_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.dim = 5  # Dimension of the problem

    def __call__(self, func):
        lb, ub = -5.0, 5.0  # Search space bounds
        # Initialize population
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        # Find the initial best solution
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = pop[best_idx]

        # Evolutionary loop
        evaluations = self.population_size
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation and Crossover
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

            # Adaptive Crossover Rate Adjustment
            if evaluations % 100 == 0:  # Adjust every 100 evaluations
                mean_fitness = np.mean(fitness)
                # Increase CR if progress is being made
                if mean_fitness < self.f_opt:
                    self.CR = min(self.CR + 0.1, 1.0)
                else:
                    self.CR = max(self.CR - 0.1, 0.1)

        return self.f_opt, self.x_opt
