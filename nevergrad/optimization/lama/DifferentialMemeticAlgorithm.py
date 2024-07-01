import numpy as np


class DifferentialMemeticAlgorithm:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        self.population_size = 20
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7

        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lb, self.ub)

                # Crossover
                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, population[i])

                # Local Search (Simple Hill Climbing)
                if np.random.rand() < 0.1:  # Small probability to invoke local search
                    trial = self.local_search(trial, func)

                # Selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

                # Update the best found solution
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt

    def local_search(self, x, func):
        step_size = 0.1
        best_x = x.copy()
        best_f = func(x)

        for i in range(self.dim):
            x_new = x.copy()
            x_new[i] += step_size * (np.random.rand() * 2 - 1)  # Small random perturbation
            x_new = np.clip(x_new, self.lb, self.ub)
            f_new = func(x_new)

            if f_new < best_f:
                best_x = x_new
                best_f = f_new

        return best_x
