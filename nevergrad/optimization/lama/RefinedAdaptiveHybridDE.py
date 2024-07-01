import numpy as np


class RefinedAdaptiveHybridDE:
    def __init__(self, budget=10000, population_size=100, F_base=0.8, CR_base=0.9, strategy_switch=0.1):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Base differential weight
        self.CR_base = CR_base  # Base crossover probability
        self.strategy_switch = strategy_switch  # Threshold to switch strategies
        self.dim = 5  # Dimensionality fixed to 5 according to the problem statement
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Initialize population randomly across the search space
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        # Initialize adaptive strategy components
        F_adaptive = np.full(self.population_size, self.F_base)
        CR_adaptive = np.full(self.population_size, self.CR_base)

        while evaluations < self.budget:
            best_idx = np.argmin(fitness)
            for i in range(self.population_size):
                # Adaptive strategy: update F and CR per individual
                F_adaptive[i] = max(0.1, self.F_base * np.exp(-4.0 * evaluations / self.budget))
                CR_adaptive[i] = self.CR_base / (1 + np.exp(-10 * (evaluations / self.budget - 0.5)))

                # Select mutation strategy dynamically
                if np.random.rand() < self.strategy_switch:
                    idxs = np.delete(np.arange(self.population_size), i)
                    a, b, c = np.random.choice(idxs, 3, replace=False)
                    mutant = np.clip(
                        population[c] + F_adaptive[i] * (population[a] - population[b]), self.lb, self.ub
                    )
                else:
                    a, b = np.random.choice(np.delete(np.arange(self.population_size), i), 2, replace=False)
                    mutant = np.clip(
                        population[i]
                        + F_adaptive[i] * (population[best_idx] - population[i])
                        + F_adaptive[i] * (population[a] - population[b]),
                        self.lb,
                        self.ub,
                    )

                # Crossover
                trial = np.where(np.random.rand(self.dim) < CR_adaptive[i], mutant, population[i])

                # Selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < fitness[best_idx]:
                        best_idx = i

                if evaluations >= self.budget:
                    break

        return fitness[best_idx], population[best_idx]
