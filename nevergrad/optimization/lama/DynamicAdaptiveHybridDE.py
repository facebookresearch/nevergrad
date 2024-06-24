import numpy as np


class DynamicAdaptiveHybridDE:
    def __init__(
        self,
        budget=10000,
        population_size=150,
        F_base=0.6,
        F_range=0.2,
        CR=0.8,
        top_fraction=0.2,
        randomization_factor=0.05,
    ):
        self.budget = budget
        self.population_size = population_size
        self.F_base = F_base  # Base Differential weight
        self.F_range = F_range  # Range for random adjustment of F
        self.CR = CR  # Crossover probability
        self.top_fraction = top_fraction  # Top fraction for elite strategy
        self.randomization_factor = randomization_factor  # Randomization factor for mutation strategy
        self.dim = 5  # Dimensionality of the problem
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Initialize population randomly
        population = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx]
        evaluations = self.population_size

        # Main loop
        while evaluations < self.budget:
            for i in range(self.population_size):
                elite_size = max(1, int(self.population_size * self.top_fraction))
                elite_indices = np.argsort(fitness)[:elite_size]

                # Mutation strategy proportionate to fitness
                if np.random.rand() < self.randomization_factor * best_fitness:
                    # More inclined to use global best mutation strategy
                    base = best_individual
                else:
                    base = population[elite_indices[np.random.randint(elite_size)]]

                # Adjust F dynamically with a slight random factor
                F = self.F_base + self.F_range * (2 * np.random.rand() - 1)

                # DE mutation
                idxs = [idx for idx in range(self.population_size) if idx not in [i, elite_indices[0]]]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                mutant = np.clip(base + F * (a - b), self.lb, self.ub)

                # Binomial crossover
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
                        best_individual = trial

                # Check budget
                if evaluations >= self.budget:
                    break

        return best_fitness, best_individual
