import numpy as np


class HybridAdaptiveOptimization:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        self.population_size = 50  # Adjusted population size for balance
        self.mutation_factor = 0.5
        self.crossover_rate = 0.9
        self.local_search_prob = 0.5  # Higher probability for local search

        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        evaluations = self.population_size

        while evaluations < self.budget:
            new_population = []
            for i in range(self.population_size):
                # Mutation (Differential Evolution)
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.mutation_factor * (b - c), self.lb, self.ub)

                # Crossover
                crossover = np.random.rand(self.dim) < self.crossover_rate
                trial = np.where(crossover, mutant, population[i])

                # Apply different local search strategies
                if np.random.rand() < self.local_search_prob:
                    trial = self.local_search(trial, func, strategy="hybrid")

                # Selection
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

    def local_search(self, x, func, strategy="hybrid"):
        best_x = x.copy()
        best_f = func(x)

        if strategy == "hybrid":
            for _ in range(5):  # Hybrid strategy with limited iterations
                # Hill climbing
                for i in range(self.dim):
                    x_new = best_x.copy()
                    step_size = 0.1 * (np.random.rand() * 2 - 1)  # Small random perturbation
                    x_new[i] = np.clip(best_x[i] + step_size, self.lb, self.ub)
                    f_new = func(x_new)

                    if f_new < best_f:
                        best_x = x_new
                        best_f = f_new

                # Gaussian mutation
                sigma = 0.1  # Standard deviation for Gaussian mutation
                x_new = best_x + np.random.normal(0, sigma, self.dim)
                x_new = np.clip(x_new, self.lb, self.ub)
                f_new = func(x_new)

                if f_new < best_f:
                    best_x = x_new
                    best_f = f_new

        return best_x
