import numpy as np


class HyperOptimizedMultiStrategicEvolutionaryOptimizerV47:
    def __init__(
        self,
        budget=10000,
        population_size=150,
        F_base=0.6,
        F_range=0.4,
        CR=0.93,
        elite_fraction=0.05,
        mutation_strategy="hybrid",
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Slightly increased mutation factor
        self.F_range = F_range  # Reduced mutation range for precision
        self.CR = CR  # Crossover probability fine-tuned
        self.elite_fraction = elite_fraction  # Decreased elite fraction to maintain diversity
        self.mutation_strategy = mutation_strategy  # Introducing a hybrid mutation strategy
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
                if self.mutation_strategy == "hybrid":
                    # Hybrid strategy: choose base from random elite or best with varying probability
                    if np.random.rand() < 0.85:
                        base = best_individual
                    else:
                        base = population[np.random.choice(elite_indices)]
                else:
                    # Default to using an elite individual as a base
                    base = population[np.random.choice(elite_indices)]

                # Mutation factor dynamic adjustment
                F = self.F_base + (np.random.rand() * self.F_range - self.F_range / 2)

                # DE/rand/1 mutation strategy
                idxs = [idx for idx in range(self.population_size) if idx not in [i, best_idx]]
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
