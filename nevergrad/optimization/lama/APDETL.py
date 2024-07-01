import numpy as np


class APDETL:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=100,
        F_base=0.5,
        CR_base=0.8,
        mutation_strategy="best",
        adaptive=True,
    ):
        self.budget = budget
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.F_base = F_base
        self.CR_base = CR_base
        self.mutation_strategy = mutation_strategy
        self.adaptive = adaptive

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dimension)
        )
        fitness = np.array([func(x) for x in population])
        evaluations = self.population_size

        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]

        while evaluations < self.budget:
            F_adaptive = self.F_base
            CR_adaptive = self.CR_base

            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Mutation
                if self.mutation_strategy == "best":
                    indices = [idx for idx in range(self.population_size) if idx != i]
                    chosen_indices = np.random.choice(indices, 2, replace=False)
                    x1, x2 = population[chosen_indices]
                    mutant = best_individual + F_adaptive * (x1 - x2)
                else:
                    indices = [idx for idx in range(self.population_size) if idx != i]
                    chosen_indices = np.random.choice(indices, 3, replace=False)
                    x0, x1, x2 = population[chosen_indices]
                    mutant = x0 + F_adaptive * (x1 - x2)

                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                cross_points = np.random.rand(self.dimension) < CR_adaptive
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dimension)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    population[i] = trial
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_individual = trial.copy()

                # Adaptive updates
                if self.adaptive:
                    successful = trial_fitness < fitness[i]
                    F_adaptive += 0.1 * (successful - 0.5)
                    CR_adaptive += 0.1 * (successful - 0.5)
                    F_adaptive = np.clip(F_adaptive, 0.1, 0.9)
                    CR_adaptive = np.clip(CR_adaptive, 0.1, 0.9)

        return best_fitness, best_individual
