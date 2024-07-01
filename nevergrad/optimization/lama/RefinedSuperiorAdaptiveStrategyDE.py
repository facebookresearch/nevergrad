import numpy as np


class RefinedSuperiorAdaptiveStrategyDE:
    def __init__(
        self, budget=10000, population_size=150, F_base=0.5, F_range=0.3, CR=0.95, strategy="refined_adaptive"
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Base Differential weight
        self.F_range = F_range  # Range to vary F for diversity
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
                # Refined adaptive mutation strategy
                if self.strategy == "refined_adaptive":
                    # Use a mix of top performers and diversity
                    sorted_indices = np.argsort(fitness)
                    high_performers = sorted_indices[: max(2, self.population_size // 10)]
                    diverse_selection = np.random.choice(high_performers, size=1)
                    base = population[diverse_selection[0]]
                else:
                    base = population[
                        np.random.choice([idx for idx in range(self.population_size) if idx != i])
                    ]

                # Adjust F dynamically with a focus on convergence
                F = self.F_base + np.random.rand() * self.F_range

                # Mutation using three random distinct indices
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(base + F * (a - b + c), self.lb, self.ub)

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
