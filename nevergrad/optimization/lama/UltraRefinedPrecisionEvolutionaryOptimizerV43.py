import numpy as np


class UltraRefinedPrecisionEvolutionaryOptimizerV43:
    def __init__(
        self,
        budget=10000,
        population_size=135,
        F_base=0.53,
        F_range=0.47,
        CR=0.93,
        elite_fraction=0.11,
        mutation_strategy="adaptive",
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Base mutation factor slightly adjusted
        self.F_range = F_range  # Slightly wider range for mutation factor adjustment
        self.CR = CR  # Adjusted crossover probability for balanced exploration and exploitation
        self.elite_fraction = elite_fraction  # Adjusted elite fraction for a more effective elite influence
        self.mutation_strategy = (
            mutation_strategy  # Adaptive mutation strategy to dynamically adjust behavior
        )
        self.dim = 5  # Dimensionality of the problem
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Initialize population uniformly within bounds
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx]

        # Main optimization loop
        while evaluations < self.budget:
            elite_size = int(self.elite_fraction * self.population_size)
            elite_indices = np.argsort(fitness)[:elite_size]

            for i in range(self.population_size):
                if self.mutation_strategy == "adaptive":
                    # Higher probability to select the best individual for mutation base
                    if np.random.rand() < 0.8:  # Increased probability to emphasize exploitation
                        base = best_individual
                    else:
                        base = population[np.random.choice(elite_indices)]
                else:
                    base = population[np.random.choice(elite_indices)]

                # Dynamically adjusted mutation factor
                F = self.F_base + (np.random.rand() * self.F_range)

                # DE/rand/1 mutation strategy
                idxs = [idx for idx in range(self.population_size) if idx not in [i, best_idx]]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(base + F * (a - b), self.lb, self.ub)

                # Binomial crossover with a slightly adjusted CR
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection based on fitness evaluation
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_individual = trial

                # Exhaustion of budget check
                if evaluations >= self.budget:
                    break

        return best_fitness, best_individual
