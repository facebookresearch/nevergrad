import numpy as np


class HybridDifferentialLocalSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.pop_size = 20
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.local_search_prob = 0.1

    def local_search(self, x, func):
        """Perform a simple local search around x"""
        steps = np.random.uniform(-0.1, 0.1, size=x.shape)
        new_x = x + steps
        new_x = np.clip(new_x, *self.bounds)
        return new_x

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None

        # Initialize population
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size

        while evaluations < self.budget:
            for i in range(self.pop_size):
                # Select three distinct individuals (but different from i)
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                # Mutation and Crossover (Differential Evolution)
                mutant = np.clip(a + self.F * (b - c), *self.bounds)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Local Search
                if np.random.rand() < self.local_search_prob:
                    trial = self.local_search(trial, func)

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
