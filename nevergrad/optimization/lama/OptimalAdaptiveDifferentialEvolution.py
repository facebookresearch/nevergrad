import numpy as np


class OptimalAdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, population_size=100, F_base=0.5, CR=0.9, strategy="elite"):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Base Differential weight
        self.CR = CR  # Crossover probability
        self.strategy = strategy  # Strategy for mutation and selection
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
                # Elite strategy: always use the best individual as the base
                if self.strategy == "elite":
                    base_idx = np.argmin(fitness)
                    base = population[base_idx]
                else:
                    base = population[
                        np.random.choice([idx for idx in range(self.population_size) if idx != i])
                    ]

                # Adjust F dynamically based on fitness values
                F = self.F_base * (1 - fitness[i] / (best_fitness + 1e-6))

                # Mutation using two random indices
                idxs = [idx for idx in range(self.population_size) if idx != i and idx != base_idx]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(base + F * (b - a), self.lb, self.ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_individual = trial

                # Exit if budget exhausted
                if evaluations >= self.budget:
                    break

        return best_fitness, best_individual
