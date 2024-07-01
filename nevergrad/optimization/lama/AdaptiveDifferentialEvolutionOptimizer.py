import numpy as np


class AdaptiveDifferentialEvolutionOptimizer:
    def __init__(self, budget=10000, pop_size=50, F=0.8, CR=0.9, adapt_factor=0.99):
        self.budget = budget
        self.pop_size = pop_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.adapt_factor = adapt_factor  # Factor for adapting F and CR
        self.dim = 5  # As stated, dimensionality is 5
        self.bounds = (-5.0, 5.0)  # Bounds are given as [-5.0, 5.0]

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.eval_count = self.pop_size

        # Evolutionary loop
        while self.eval_count < self.budget:
            for i in range(self.pop_size):
                # Mutation
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.bounds[0], self.bounds[1])

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f_trial = func(trial)
                self.eval_count += 1
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial

                if self.eval_count >= self.budget:
                    break

            # Adapt F and CR
            self.F *= self.adapt_factor
            self.CR *= self.adapt_factor

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        return self.f_opt, self.x_opt
