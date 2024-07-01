import numpy as np


class OptimizedUltraRefinedPrecisionEvolutionaryOptimizerV45:
    def __init__(
        self,
        budget=10000,
        population_size=120,
        F_base=0.6,
        F_range=0.3,
        CR=0.9,
        elite_fraction=0.08,
        mutation_strategy="adaptive",
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Modest increase in the base mutation factor for more explorative mutations
        self.F_range = F_range  # Reduced mutation range to focus on more precise tuning
        self.CR = CR  # Slightly reduced crossover probability to balance exploration and exploitation
        self.elite_fraction = elite_fraction  # Reduced elite fraction to increase diversity in the population
        self.mutation_strategy = mutation_strategy  # Retain adaptive mutation strategy with enhancements
        self.dim = 5  # Dimensionality of the problem remains constant
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
                    # Dynamically choose the base individual from the elite pool or the best so far
                    if (
                        np.random.rand() < 0.85
                    ):  # Increased emphasis on the best individual to promote exploitation
                        base = best_individual
                    else:
                        base = population[np.random.choice(elite_indices)]
                else:
                    base = population[np.random.choice(elite_indices)]

                # Dynamic mutation factor based on current stage of optimization
                F = self.F_base + (np.random.rand() * self.F_range)

                # Mutation strategy DE/rand/1
                idxs = [idx for idx in range(self.population_size) if idx not in [i, best_idx]]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(base + F * (a - b), self.lb, self.ub)

                # Binomial crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] is True
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate the trial solution
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_individual = trial

                # Terminate if the budget is exhausted
                if evaluations >= self.budget:
                    break

        return best_fitness, best_individual
