import numpy as np


class OptimizedHybridStrategyDE:
    def __init__(self, budget=10000, population_size=150, F=0.85, CR=0.95, adaptive=True):
        self.budget = budget
        self.population_size = population_size
        self.F = F  # Differential weight, potentially adaptive
        self.CR = CR  # Crossover probability
        self.adaptive = adaptive  # Enable adaptive control of parameters
        self.dim = 5  # Dimensionality of the problem
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Initialize population uniformly
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]

        # Adaptation coefficients
        F_l, F_u = 0.6, 1.0  # Lower and upper bounds for F
        CR_l, CR_u = 0.85, 1.0  # Adaptive bounds for CR

        evaluations = self.population_size
        while evaluations < self.budget:
            for i in range(self.population_size):
                if self.adaptive:
                    # Adaptive F and CR based on normalized evaluations
                    self.F = F_l + (F_u - F_l) * np.exp(-4.0 * (self.budget - evaluations) / self.budget)
                    self.CR = CR_l + (CR_u - CR_l) * np.exp(-4.0 * evaluations / self.budget)

                # Mutation: DE/rand/1 with possible best strategy switch
                if np.random.rand() < 0.5:  # 50% chance to switch strategy
                    idxs = np.arange(self.population_size)
                    idxs = np.delete(idxs, i)
                    a, b, c = np.random.choice(idxs, 3, replace=False)
                    mutant = np.clip(
                        population[c] + self.F * (population[a] - population[b]), self.lb, self.ub
                    )
                else:
                    best_idx = np.argmin(fitness)  # Always find the current best
                    idxs = np.arange(self.population_size)
                    idxs = np.delete(idxs, i)
                    a, b = np.random.choice(idxs, 2, replace=False)
                    mutant = np.clip(
                        population[i]
                        + self.F * (best_individual - population[i])
                        + self.F * (population[a] - population[b]),
                        self.lb,
                        self.ub,
                    )

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
                        best_individual = trial

                if evaluations >= self.budget:
                    break

        return fitness[best_idx], best_individual
