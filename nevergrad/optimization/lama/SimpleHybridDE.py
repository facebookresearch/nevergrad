import numpy as np


class SimpleHybridDE:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.pop_size = 20
        self.F = 0.7  # Differential weight
        self.CR = 0.9  # Crossover probability

    def random_bounds(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], self.dim)

    def local_search(self, x, func):
        best = x
        f_best = func(x)
        perturbations = np.linspace(-0.1, 0.1, 5)
        for d in range(self.dim):
            for perturb in perturbations:
                x_new = np.copy(x)
                x_new[d] += perturb
                x_new = np.clip(x_new, self.bounds[0], self.bounds[1])
                f_new = func(x_new)
                if f_new < f_best:
                    best = x_new
                    f_best = f_new
        return best

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        # Initialize population
        population = np.array([self.random_bounds() for _ in range(self.pop_size)])
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size

        while evaluations < self.budget:
            for i in range(self.pop_size):
                # Select three distinct individuals (but different from i)
                indices = np.arange(self.pop_size)
                indices = indices[indices != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Differential Evolution mutation and crossover
                mutant = np.clip(a + self.F * (b - c), self.bounds[0], self.bounds[1])
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Local Search with a small probability
                if np.random.rand() < 0.2 and evaluations + 5 <= self.budget:
                    trial = self.local_search(trial, func)
                    evaluations += 5

                # Selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                # Check if we've exhausted our budget
                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt
