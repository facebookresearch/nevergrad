import numpy as np


class PGDE:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=100,
        F_base=0.8,
        CR_base=0.8,
        adaptivity_factor=0.1,
    ):
        self.budget = budget
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.F_base = F_base
        self.CR_base = CR_base
        self.adaptivity_factor = adaptivity_factor

    def __call__(self, func):
        # Initialize population uniformly
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dimension)
        )
        fitness = np.array([func(x) for x in population])
        evaluations = self.population_size

        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]

        # Initialize adaptation parameters for F and CR
        F = np.full(self.population_size, self.F_base)
        CR = np.full(self.population_size, self.CR_base)

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Differential mutation using "rand/1" strategy
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = population[a] + F[i] * (population[b] - population[c])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Binomial Crossover
                cross_points = np.random.rand(self.dimension) < CR[i]
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dimension)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()

                # Adaptive mechanism for exploration and exploitation
                F[i] += (
                    self.adaptivity_factor * (trial_fitness < fitness[i]) * (F[i] - 0.1) * np.random.randn()
                )
                CR[i] += (
                    self.adaptivity_factor * (trial_fitness < fitness[i]) * (CR[i] - 0.1) * np.random.randn()
                )
                F[i] = max(0.1, min(F[i], 1.0))
                CR[i] = max(0.1, min(CR[i], 1.0))

        return best_fitness, best_individual
