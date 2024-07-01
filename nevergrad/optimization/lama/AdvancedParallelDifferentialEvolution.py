import numpy as np


class AdvancedParallelDifferentialEvolution:
    def __init__(self, budget=10000, population_size=100, F=0.8, CR=0.9, strategy="rand/1/bin"):
        self.budget = budget
        self.population_size = population_size
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability
        self.strategy = strategy  # Mutation and crossover strategy
        self.dim = 5  # Dimensionality of the problem
        self.lb = -5.0  # Lower bound of the search space
        self.ub = 5.0  # Upper bound of the search space

    def __call__(self, func):
        # Initialize population randomly
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation: different strategies can be implemented
                if self.strategy == "best/1/bin":
                    best_idx = np.argmin(fitness)
                    base = population[best_idx]
                elif self.strategy == "rand/1/bin":
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    base = population[np.random.choice(idxs)]
                else:
                    raise ValueError("Unsupported strategy")

                # DE/rand/1/bin: mutation and crossover
                idxs = np.delete(np.arange(self.population_size), i)
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(base + self.F * (a - b), self.lb, self.ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, population[i])

                # Evaluation and selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

                if evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return fitness[best_idx], population[best_idx]
