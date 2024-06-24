import numpy as np


class HyperOptimizedMultiStrategicEvolutionaryOptimizerV48:
    def __init__(
        self,
        budget=10000,
        population_size=140,
        F_base=0.58,
        F_range=0.42,
        CR=0.92,
        elite_fraction=0.08,
        mutation_strategy="adaptive",
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Slightly increased base mutation factor for stronger exploration
        self.F_range = F_range  # Adjusted mutation range for controlled exploration
        self.CR = CR  # Adjusted crossover probability for better exploitation
        self.elite_fraction = (
            elite_fraction  # Adjusted elite fraction to balance exploration and exploitation
        )
        self.mutation_strategy = mutation_strategy  # Employing adaptive mutation strategy
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
                    # Adaptive strategy: choose between best or random elite based on adaptive probability
                    if np.random.rand() < 0.8:
                        base = best_individual
                    else:
                        base = population[np.random.choice(elite_indices)]
                else:
                    base = population[np.random.choice(elite_indices)]

                # Mutation factor dynamic adjustment for more aggressive exploration
                F = self.F_base + (np.random.rand() * self.F_range - self.F_range / 2)

                # DE/rand/1 mutation strategy
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(base + F * (a - b), self.lb, self.ub)

                # Crossover operation
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Evaluation and selection
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
