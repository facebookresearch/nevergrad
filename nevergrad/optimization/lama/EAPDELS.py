import numpy as np


class EAPDELS:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=100,
        F_base=0.6,
        CR_base=0.9,
        mutation_strategy="rand/1/bin",
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

        # Adaptive parameters initialization
        F = np.full(self.population_size, self.F_base)
        CR = np.full(self.population_size, self.CR_base)

        while evaluations < self.budget:
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                # Mutation (DE/rand/1/bin strategy)
                indices = [idx for idx in range(self.population_size) if idx != i]
                chosen_indices = np.random.choice(indices, 3, replace=False)
                x0, x1, x2 = population[chosen_indices]
                mutant = x0 + F[i] * (x1 - x2)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover (binomial)
                cross_points = np.random.rand(self.dimension) < CR[i]
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
                        # Intensification: Local Search
                        local_search_step = 0.1 * (self.upper_bound - self.lower_bound)
                        local_candidate = best_individual + np.random.uniform(
                            -local_search_step, local_search_step, self.dimension
                        )
                        local_candidate = np.clip(local_candidate, self.lower_bound, self.upper_bound)
                        local_fitness = func(local_candidate)
                        evaluations += 1
                        if local_fitness < best_fitness:
                            best_fitness = local_fitness
                            best_individual = local_candidate

                # Adaptive parameter update
                if self.adaptive:
                    success = trial_fitness < fitness[i]
                    F[i] += 0.05 * (success - 0.5)
                    CR[i] += 0.05 * (success - 0.5)
                    F[i] = np.clip(F[i], 0.1, 0.9)
                    CR[i] = np.clip(CR[i], 0.1, 0.9)

        return best_fitness, best_individual
