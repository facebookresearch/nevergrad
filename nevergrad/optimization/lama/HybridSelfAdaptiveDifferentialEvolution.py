import numpy as np


class HybridSelfAdaptiveDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

        self.population_size = 50
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.T = 10  # Local search iterations

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        evaluations = self.population_size

        while evaluations < self.budget:
            new_population = []
            for i in range(self.population_size):
                # Mutation step
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lb, self.ub)

                # Crossover step
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[i])

                # Self-adaptive local search strategy
                if np.random.rand() < 0.5:
                    trial = self.local_search(trial, func)

                # Selection step
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    new_population.append(trial)
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial
                else:
                    new_population.append(population[i])

                if evaluations >= self.budget:
                    break

            population = np.array(new_population)
            fitness = np.array([func(ind) for ind in population])

        return self.f_opt, self.x_opt

    def local_search(self, x, func):
        best_x = x.copy()
        best_f = func(x)

        for _ in range(self.T):
            for i in range(self.dim):
                x_new = best_x.copy()
                step_size = np.random.uniform(-0.1, 0.1)
                x_new[i] = np.clip(best_x[i] + step_size, self.lb, self.ub)
                f_new = func(x_new)

                if f_new < best_f:
                    best_x = x_new
                    best_f = f_new

        return best_x

    def adaptive_F_CR(self, i):
        # Adaptive parameters adjustment
        if i % 100 == 0:
            self.F = np.random.uniform(0.4, 0.9)
            self.CR = np.random.uniform(0.1, 0.9)
