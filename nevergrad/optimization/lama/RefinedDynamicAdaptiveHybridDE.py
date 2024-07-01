import numpy as np


class RefinedDynamicAdaptiveHybridDE:
    def __init__(
        self,
        budget=10000,
        population_size=100,
        F_base=0.5,
        F_range=0.4,
        CR=0.85,
        elite_fraction=0.1,
        mutation_factor=0.1,
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Base Differential weight
        self.F_range = F_range  # Dynamic range for the mutation factor F
        self.CR = CR  # Crossover probability
        self.elite_fraction = elite_fraction  # Fraction of top performers considered elite
        self.mutation_factor = mutation_factor  # Proportion of randomization in mutation
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
            elite_size = int(self.elite_fraction * self.population_size)
            elite_indices = np.argsort(fitness)[:elite_size]

            for i in range(self.population_size):
                # Mutation strategy using random elite selection
                if np.random.rand() < self.mutation_factor:
                    base = population[elite_indices[np.random.randint(elite_size)]]
                else:
                    # More likely to use the best individual
                    base = best_individual

                # Adjust F dynamically
                F = self.F_base + (2 * np.random.rand() - 1) * self.F_range

                # DE mutation
                idxs = [idx for idx in range(self.population_size) if idx not in [i, best_idx]]
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
                        best_fitness = f_trial
                        best_individual = trial

                # Exit if budget exhausted
                if evaluations >= self.budget:
                    break

        return best_fitness, best_individual
