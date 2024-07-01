import numpy as np


class UltraRefinedHyperStrategicOptimizerV54:
    def __init__(
        self,
        budget=10000,
        population_size=120,
        F_base=0.60,
        F_range=0.40,
        CR=0.93,
        elite_fraction=0.08,
        mutation_strategy="adaptive",
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Slightly increased base mutation factor for stronger mutations
        self.F_range = F_range  # Narrowed mutation range for controlled exploration
        self.CR = CR  # Adjusted crossover probability for a better balance
        self.elite_fraction = (
            elite_fraction  # Optimized elite fraction to focus on a narrower set of top individuals
        )
        self.mutation_strategy = mutation_strategy  # Mutation strategy remains adaptive for flexibility
        self.dim = 5  # Problem dimensionality
        self.lb = -5.0  # Lower bound of the search space
        self.ub = 5.0  # Upper bound of the search space

    def __call__(self, func):
        # Initialize population within the search space bounds
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
                # Adaptive mutation strategy: higher chance to pick the best individual
                if self.mutation_strategy == "adaptive":
                    if np.random.rand() < 0.85:  # Increased probability to focus more on the current best
                        base = best_individual
                    else:
                        base = population[np.random.choice(elite_indices)]
                else:
                    base = population[np.random.choice(elite_indices)]

                # Adjust mutation factor dynamically
                F = self.F_base + (2 * np.random.rand() - 1) * self.F_range

                # Mutation using DE/rand/1/bin scheme
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(base + F * (a - b), self.lb, self.ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Fitness evaluation and selection
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
