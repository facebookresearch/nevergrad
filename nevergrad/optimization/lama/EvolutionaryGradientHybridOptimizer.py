import numpy as np


class EvolutionaryGradientHybridOptimizer:
    def __init__(
        self, budget=10000, population_size=50, F_base=0.6, F_increment=0.2, CR=0.9, elite_fraction=0.2
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Base Differential weight
        self.F_increment = F_increment  # Increment for the mutation factor F
        self.CR = CR  # Crossover probability
        self.elite_fraction = elite_fraction  # Fraction of top performers considered elite
        self.dim = 5  # Dimensionality of the problem
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Initialize population randomly
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()

        # Main loop
        while evaluations < self.budget:
            F = self.F_base + np.sin(evaluations / self.budget * np.pi) * self.F_increment

            elite_size = int(self.elite_fraction * self.population_size)
            elite_indices = np.argsort(fitness)[:elite_size]

            for i in range(self.population_size):
                # Choose three random indices excluding current index i
                candidates = np.random.choice(elite_indices, 3, replace=False)
                x1, x2, x3 = population[candidates[0]], population[candidates[1]], population[candidates[2]]

                # Mutation (DE/rand/1/bin)
                mutant = x1 + F * (x2 - x3)
                mutant = np.clip(mutant, self.lb, self.ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f_trial = func(trial)
                evaluations += 1

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_individual = trial.copy()

                # Check budget
                if evaluations >= self.budget:
                    break

        return best_fitness, best_individual
