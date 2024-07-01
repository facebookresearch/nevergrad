import numpy as np


class AdvancedDiversityDE:
    def __init__(self, budget=10000, population_size=100, F_init=0.5, CR=0.8, learning_rate=0.1):
        self.budget = budget
        self.population_size = population_size
        self.F = F_init  # Initial Differential weight
        self.CR = CR  # Crossover probability
        self.learning_rate = learning_rate  # Learning rate for adaptive F
        self.dim = 5  # Dimensionality of the problem
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Initialize population randomly
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        best_idx = np.argmin(fitness)
        best = population[best_idx]

        # Main loop
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Adaptive F update
                self.F *= (1 - self.learning_rate) + self.learning_rate * np.random.normal(loc=0.5, scale=0.1)

                # Mutation using "best" individual
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(best + self.F * (a - b), self.lb, self.ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < fitness[best_idx]:
                        best_idx = i
                        best = trial

                # Exit if budget exhausted
                if evaluations >= self.budget:
                    break

        # Find and return the best solution
        return fitness[best_idx], population[best_idx]
