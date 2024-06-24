import numpy as np


class UltraRefinedHyperStrategicOptimizerV50:
    def __init__(
        self,
        budget=10000,
        population_size=150,
        F_base=0.6,
        F_range=0.3,
        CR=0.92,
        elite_fraction=0.08,
        mutation_strategy="adaptive",
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Adjusted base mutation factor for more stable convergence
        self.F_range = F_range  # Narrower range for mutation factor to control exploration dynamically
        self.CR = CR  # Slightly reduced crossover probability to allow better exploitation
        self.elite_fraction = elite_fraction  # Reduced elite fraction to intensify competition
        self.mutation_strategy = mutation_strategy  # Maintained adaptive strategy for flexibility
        self.dim = 5  # Dimensionality of the problem
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Initialize population within the given bounds
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
                # Select base individual based on strategy
                if self.mutation_strategy == "adaptive":
                    if np.random.rand() < 0.8:  # Increased probability to choose the current best
                        base = best_individual
                    else:
                        base = population[np.random.choice(elite_indices)]
                else:
                    base = population[np.random.choice(elite_indices)]

                # Adjust F dynamically within a narrower range
                F = self.F_base + np.random.normal(0, self.F_range / 2)

                # Mutation using DE/rand/1/bin
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(base + F * (a - b), self.lb, self.ub)

                # Crossover operation
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

                # Break the loop if the budget is reached
                if evaluations >= self.budget:
                    break

        return best_fitness, best_individual
