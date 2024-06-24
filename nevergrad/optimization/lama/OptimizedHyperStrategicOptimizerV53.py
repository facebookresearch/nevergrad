import numpy as np


class OptimizedHyperStrategicOptimizerV53:
    def __init__(
        self,
        budget=10000,
        population_size=150,
        F_base=0.58,
        F_range=0.42,
        CR=0.92,
        elite_fraction=0.06,
        mutation_strategy="adaptive",
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Increased base mutation factor for enhanced exploration
        self.F_range = F_range  # Slightly narrowed mutation factor range for more controlled mutations
        self.CR = CR  # Adjusted Crossover probability to facilitate better exploration-exploitation balance
        self.elite_fraction = elite_fraction  # Reduced elite fraction to sharpen focus on the top performers
        self.mutation_strategy = mutation_strategy  # Mutation strategy remains adaptive
        self.dim = 5  # Fixed dimension
        self.lb = -5.0  # Search space lower bound
        self.ub = 5.0  # Search space upper bound

    def __call__(self, func):
        # Initialize population uniformly within bounds
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx]

        # Optimization main loop
        while evaluations < self.budget:
            elite_size = int(self.elite_fraction * self.population_size)
            elite_indices = np.argsort(fitness)[:elite_size]

            for i in range(self.population_size):
                if self.mutation_strategy == "adaptive":
                    # Adaptive strategy for base selection using a higher probability for the best individual
                    if np.random.rand() < 0.8:
                        base = best_individual
                    else:
                        base = population[np.random.choice(elite_indices)]
                else:
                    # Random elite base if not adaptive
                    base = population[np.random.choice(elite_indices)]

                # Dynamic F adjustment
                F = self.F_base + (2 * np.random.rand() - 1) * self.F_range

                # Mutation using DE/rand/1/bin scheme
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(base + F * (a - b), self.lb, self.ub)

                # Binomial crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection based on fitness comparison
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_individual = trial

                # Exit loop if budget exceeded
                if evaluations >= self.budget:
                    break

        return best_fitness, best_individual
