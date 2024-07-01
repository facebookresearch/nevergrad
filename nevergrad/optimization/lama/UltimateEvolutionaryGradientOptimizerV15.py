import numpy as np


class UltimateEvolutionaryGradientOptimizerV15:
    def __init__(
        self,
        budget=10000,
        population_size=140,
        F_base=0.58,
        F_range=0.42,
        CR=0.92,
        elite_fraction=0.08,
        mutation_strategy="balanced",
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Base mutation factor
        self.F_range = F_range  # Dynamic range for mutation factor adjustment
        self.CR = CR  # Crossover probability
        self.elite_fraction = elite_fraction  # Fraction of top performers considered elite
        self.mutation_strategy = (
            mutation_strategy  # Type of mutation strategy: 'adaptive', 'random', 'balanced'
        )
        self.dim = 5  # Dimensionality of the problem
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
                # Mutation strategy selection
                if self.mutation_strategy == "adaptive":
                    base = (
                        best_individual
                        if np.random.rand() < 0.75
                        else population[np.random.choice(elite_indices)]
                    )
                elif self.mutation_strategy == "random":
                    base = population[np.random.choice(elite_indices)]
                elif self.mutation_strategy == "balanced":
                    if np.random.rand() < 0.5:
                        base = best_individual
                    else:
                        base = population[np.random.choice(elite_indices)]

                # Dynamic adjustment of F
                F = self.F_base + (2 * np.random.rand() - 1) * self.F_range

                # DE/rand/1 mutation
                idxs = [idx for idx in range(self.population_size) if idx not in [i, best_idx]]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(base + F * (a - b), self.lb, self.ub)

                # Binomial crossover
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

                # Check if budget is exhausted
                if evaluations >= self.budget:
                    break

        return best_fitness, best_individual
