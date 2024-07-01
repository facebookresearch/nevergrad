import numpy as np


class EnhancedHyperOptimizedMultiStrategicOptimizerV49:
    def __init__(
        self,
        budget=10000,
        population_size=150,
        F_base=0.5,
        F_range=0.4,
        CR=0.9,
        elite_fraction=0.05,
        mutation_strategy="adaptive",
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Base mutation factor, slightly reduced for more stable exploration
        self.F_range = F_range  # Adjusted mutation factor range for controlled exploration
        self.CR = CR  # Crossover probability, slightly reduced to promote more exploitation
        self.elite_fraction = elite_fraction  # Reduced elite fraction to increase competitive pressure
        self.mutation_strategy = mutation_strategy  # Adaptive mutation strategy
        self.dim = 5  # Dimensionality of the problem
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Initialize population within bounds
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
                    # Enhanced adaptive strategy: prefer current best slightly more often
                    if np.random.rand() < 0.85:  # Increased probability to select the current best
                        base = best_individual
                    else:
                        base = population[np.random.choice(elite_indices)]
                else:
                    base = population[np.random.choice(elite_indices)]

                # Dynamically adjust mutation factor for stronger exploitation
                F = self.F_base + np.random.normal(0, self.F_range / 2)

                # DE/rand/1/bin mutation strategy
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(base + F * (a - b), self.lb, self.ub)

                # Crossover operation
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection based on fitness
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_individual = trial

                # Break if budget is exhausted
                if evaluations >= self.budget:
                    break

        return best_fitness, best_individual
