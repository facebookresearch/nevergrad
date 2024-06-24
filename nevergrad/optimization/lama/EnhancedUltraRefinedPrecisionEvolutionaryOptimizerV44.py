import numpy as np


class EnhancedUltraRefinedPrecisionEvolutionaryOptimizerV44:
    def __init__(
        self,
        budget=10000,
        population_size=140,
        F_base=0.58,
        F_range=0.42,
        CR=0.98,
        elite_fraction=0.09,
        mutation_strategy="adaptive",
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Slightly increased mutation base
        self.F_range = F_range  # Narrowed mutation range for more precise adjustments
        self.CR = CR  # Increased crossover probability for tighter exploration
        self.elite_fraction = elite_fraction  # Slightly decreased elite fraction for more diversity
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
                    # Use best or random elite based on a dynamic probability that focuses more on local search
                    if np.random.rand() < 0.8:  # Increased probability to emphasize elite influence
                        base = best_individual
                    else:
                        base = population[np.random.choice(elite_indices)]
                else:
                    base = population[np.random.choice(elite_indices)]

                # Dynamically adjusted mutation factor
                F = self.F_base + (np.random.rand() * self.F_range)

                # DE/rand/1 mutation strategy with dynamic mutation factor
                idxs = [idx for idx in range(self.population_size) if idx not in [i, best_idx]]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(base + F * (a - b), self.lb, self.ub)

                # Binomial crossover with a high crossover rate
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
