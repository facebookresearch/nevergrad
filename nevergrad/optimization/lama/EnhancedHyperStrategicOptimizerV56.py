import numpy as np


class EnhancedHyperStrategicOptimizerV56:
    def __init__(
        self,
        budget=10000,
        population_size=140,
        F_base=0.57,
        F_range=0.43,
        CR=0.94,
        elite_fraction=0.15,
        mutation_strategy="adaptive",
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Slightly adjusted mutation factor for balance
        self.F_range = F_range  # Slightly narrower mutation range for stability
        self.CR = CR  # Optimized crossover probability for improved gene mixing
        self.elite_fraction = elite_fraction  # Increased elite fraction to enhance exploitation
        self.mutation_strategy = (
            mutation_strategy  # Adaptive mutation strategy for dynamic problem adaptation
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
                # Use adaptive mutation strategy with a modified selection probability
                if self.mutation_strategy == "adaptive":
                    if (
                        np.random.rand() < 0.85
                    ):  # Increased probability to select the current best, focusing search intensity
                        base = best_individual
                    else:
                        base = population[np.random.choice(elite_indices)]
                else:
                    base = population[np.random.choice(elite_indices)]

                # Dynamic adjustment of mutation factor F
                F = self.F_base + (2 * np.random.rand() - 1) * self.F_range

                # Mutation using DE/rand/1/bin scheme with tweaks
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(base + F * (a - b), self.lb, self.ub)

                # Crossover using a slightly altered CR to improve solution mixing
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate trial solution
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_individual = trial

                # Exit if budget exhausted
                if evaluations >= self.budget:
                    break

        return best_fitness, best_individual
