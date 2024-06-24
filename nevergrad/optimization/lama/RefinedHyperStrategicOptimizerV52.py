import numpy as np


class RefinedHyperStrategicOptimizerV52:
    def __init__(
        self,
        budget=10000,
        population_size=150,
        F_base=0.55,
        F_range=0.45,
        CR=0.95,
        elite_fraction=0.08,
        mutation_strategy="adaptive",
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Slightly increased base mutation factor to enhance global exploration
        self.F_range = F_range  # Maintaining a moderate mutation factor range for balanced dynamics
        self.CR = CR  # High crossover probability to ensure good trait mixing
        self.elite_fraction = (
            elite_fraction  # Reducing elite fraction to concentrate more on the very best solutions
        )
        self.mutation_strategy = mutation_strategy  # Adaptive strategy to dynamically select mutation base
        self.dim = 5  # Dimensionality of the problem
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Initialize population within the search space bounds
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx]

        # Optimization loop
        while evaluations < self.budget:
            elite_size = int(self.elite_fraction * self.population_size)
            elite_indices = np.argsort(fitness)[:elite_size]

            for i in range(self.population_size):
                if self.mutation_strategy == "adaptive":
                    # Enhance probability selection for the best individual
                    if np.random.rand() < 0.75:
                        base = best_individual
                    else:
                        base = population[np.random.choice(elite_indices)]
                else:
                    base = population[np.random.choice(elite_indices)]

                # Adjusting F dynamically within a controlled range
                F = self.F_base + (2 * np.random.rand() - 1) * self.F_range

                # Mutation using DE/rand/1/bin scheme
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(base + F * (a - b), self.lb, self.ub)

                # Crossover using binomial method
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection step to potentially update population
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_individual = trial

                # Break the loop if the budget is exceeded
                if evaluations >= self.budget:
                    break

        return best_fitness, best_individual
