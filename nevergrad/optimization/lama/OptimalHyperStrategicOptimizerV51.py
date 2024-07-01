import numpy as np


class OptimalHyperStrategicOptimizerV51:
    def __init__(
        self,
        budget=10000,
        population_size=130,
        F_base=0.50,
        F_range=0.4,
        CR=0.93,
        elite_fraction=0.05,
        mutation_strategy="adaptive",
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Adjusted base mutation factor for a balance of exploration and exploitation
        self.F_range = F_range  # Narrowed range for mutation factor to maintain controlled variability
        self.CR = CR  # Adjusted crossover probability to ensure sufficient mixing
        self.elite_fraction = elite_fraction  # Reduced elite fraction to focus on the best solutions
        self.mutation_strategy = mutation_strategy  # Utilizing adaptive strategy for dynamic adjustments
        self.dim = 5  # Dimensionality of the problem
        self.lb = -5.0  # Lower bound of the search space
        self.ub = 5.0  # Upper bound of the search space

    def __call__(self, func):
        # Initialize population within the given bounds
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
                    # Enhanced probability to select based on fitness
                    if np.random.rand() < 0.8:
                        base = best_individual
                    else:
                        base = population[np.random.choice(elite_indices)]
                else:
                    base = population[np.random.choice(elite_indices)]

                # Dynamic adjustment of F within a controlled range
                F = self.F_base + np.random.normal(0, self.F_range / 2)

                # DE/rand/1 mutation scheme
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(base + F * (a - b), self.lb, self.ub)

                # Crossover operation using binomial crossover
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

                # Exit if the budget is exceeded
                if evaluations >= self.budget:
                    break

        return best_fitness, best_individual
