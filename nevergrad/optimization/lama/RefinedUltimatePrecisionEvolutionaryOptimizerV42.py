import numpy as np


class RefinedUltimatePrecisionEvolutionaryOptimizerV42:
    def __init__(
        self,
        budget=10000,
        population_size=150,
        F_base=0.52,
        F_range=0.48,
        CR=0.92,
        elite_fraction=0.15,
        mutation_strategy="adaptive",
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Reduced base mutation factor for more stable exploration
        self.F_range = F_range  # Increased range for mutation factor to enhance adaptive capabilities
        self.CR = CR  # Slightly lower crossover probability to maintain diversity
        self.elite_fraction = (
            elite_fraction  # Increased elite fraction to improve the influence of top performers
        )
        self.mutation_strategy = (
            mutation_strategy  # Adaptive mutation strategy for dynamic behavior based on fitness landscape
        )
        self.dim = 5  # Problem dimensionality
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

        # Main optimization loop
        while evaluations < self.budget:
            elite_size = int(self.elite_fraction * self.population_size)
            elite_indices = np.argsort(fitness)[:elite_size]

            for i in range(self.population_size):
                # Adaptive base individual selection
                if self.mutation_strategy == "adaptive":
                    if np.random.rand() < 0.80:  # Higher preference for the best individual
                        base = best_individual
                    else:
                        base = population[np.random.choice(elite_indices)]
                else:
                    base = population[np.random.choice(elite_indices)]

                # Dynamically adjusted mutation factor
                F = self.F_base + (2 * np.random.rand() - 1) * self.F_range

                # DE/rand/1 mutation
                idxs = [idx for idx in range(self.population_size) if idx not in [i, best_idx]]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(base + F * (a - b), self.lb, self.ub)

                # Binomial crossover with refined CR
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
