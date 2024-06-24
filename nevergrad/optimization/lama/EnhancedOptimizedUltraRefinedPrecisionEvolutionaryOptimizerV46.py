import numpy as np


class EnhancedOptimizedUltraRefinedPrecisionEvolutionaryOptimizerV46:
    def __init__(
        self,
        budget=10000,
        population_size=140,
        F_base=0.58,
        F_range=0.42,
        CR=0.92,
        elite_fraction=0.07,
        mutation_strategy="adaptive",
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Fine-tuned mutation factor
        self.F_range = F_range  # Adjusted mutation range for better control
        self.CR = CR  # Adjusted crossover probability to enhance convergence stability
        self.elite_fraction = elite_fraction  # Adjusted elite fraction to optimize diversity and convergence
        self.mutation_strategy = mutation_strategy  # Retained adaptive mutation strategy for dynamic response
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
                    # Choose the base individual from the elite pool or the best, dynamically
                    if np.random.rand() < 0.8:  # Increased use of best individual to focus search
                        base = best_individual
                    else:
                        base = population[np.random.choice(elite_indices)]
                else:
                    base = population[np.random.choice(elite_indices)]

                # Mutation factor adjustment
                F = self.F_base + (np.random.rand() * self.F_range - self.F_range / 2)

                # Mutation strategy: DE/rand/1
                idxs = [idx for idx in range(self.population_size) if idx not in [i, best_idx]]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(base + F * (a - b), self.lb, self.ub)

                # Crossover
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

                # Termination check
                if evaluations >= self.budget:
                    break

        return best_fitness, best_individual
