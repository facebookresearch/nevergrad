import numpy as np


class OptimalEvolutionaryGradientHybridOptimizerV8:
    def __init__(
        self,
        budget=10000,
        population_size=130,
        F_base=0.59,
        F_range=0.39,
        CR=0.91,
        elite_fraction=0.13,
        mutation_strategy="adaptive",
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Base mutation factor
        self.F_range = F_range  # Dynamic range for mutation factor adjustment
        self.CR = CR  # Crossover probability
        self.elite_fraction = elite_fraction  # Fraction of top performers considered elite
        self.mutation_strategy = mutation_strategy  # Type of mutation strategy: 'adaptive' or 'fixed'
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
                if self.mutation_strategy == "adaptive":
                    # Use best or random elite based on mutation strategy
                    if np.random.rand() < 0.8:  # Increase the probability to select the current best
                        base = best_individual
                    else:
                        base = population[np.random.choice(elite_indices)]

                # Dynamic adjustment of F with a refined adjustment strategy
                F = self.F_base + (2 * np.random.rand() - 1) * self.F_range

                # DE/rand/1 mutation method
                idxs = [
                    idx
                    for idx in range(self.population_size)
                    if idx not in [i, best_idx] + list(elite_indices)
                ]
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
