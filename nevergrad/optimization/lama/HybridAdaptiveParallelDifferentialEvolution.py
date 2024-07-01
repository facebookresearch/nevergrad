import numpy as np


class HybridAdaptiveParallelDifferentialEvolution:
    def __init__(self, budget=10000, population_size=100, F=0.8, CR=0.9, adaptive=True):
        self.budget = budget
        self.population_size = population_size
        self.F = F  # Base differential weight, potentially adaptive
        self.CR = CR  # Crossover probability
        self.adaptive = adaptive  # Enable adaptive control of parameters
        self.dim = 5  # Dimensionality of the problem
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Initialize population randomly
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        # Best individual tracker
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]

        # Adaptation parameters
        F_l, F_u = 0.5, 0.9  # Lower and upper bounds for F

        # Main loop
        evaluations = self.population_size
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Adaptive F
                if self.adaptive:
                    self.F = F_l + (F_u - F_l) * np.exp(-(evaluations / self.budget))

                # Mutation: DE/current-to-best/1
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

                # Check if budget exhausted
                if evaluations >= self.budget:
                    break

        return fitness[best_idx], population[best_idx]
