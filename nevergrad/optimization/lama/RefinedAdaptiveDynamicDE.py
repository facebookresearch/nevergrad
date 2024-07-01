import numpy as np


class RefinedAdaptiveDynamicDE:
    def __init__(
        self, budget=10000, population_size=150, F_base=0.5, F_range=0.3, CR=0.9, strategy="dynamic"
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Base Differential weight
        self.F_range = F_range  # Range to vary F for diversity and adaptation
        self.CR = CR  # Crossover probability
        self.strategy = strategy  # Strategy for mutation based on the learning phase
        self.dim = 5  # Dimensionality of the problem
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Initialize population randomly
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx]

        # Main loop
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Adaptive strategy for mutation: switches based on phase of optimization
                if evaluations < self.budget * 0.5:
                    # Exploration phase: use random best from top 10%
                    best_samples = np.argsort(fitness)[: max(1, self.population_size // 10)]
                    base = population[np.random.choice(best_samples)]
                else:
                    # Exploitation phase: focus more on the current best
                    base = population[best_idx]

                # Dynamic adjustment of F based on the optimization phase
                F = self.F_base + (np.sin(evaluations / self.budget * np.pi) * self.F_range)

                # Mutation using DE/rand/1/bin strategy
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(base + F * (a - b), self.lb, self.ub)

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
                    if f_trial < best_fitness:
                        best_idx = i
                        best_fitness = f_trial
                        best_individual = trial

                # Exit if budget exhausted
                if evaluations >= self.budget:
                    break

        # Return the best solution found
        return best_fitness, best_individual
