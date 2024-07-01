import numpy as np


class CMDEALX:
    def __init__(
        self, budget, population_size=50, F_init=0.5, CR_init=0.9, local_search_factor=0.1, max_local_steps=20
    ):
        self.budget = budget
        self.CR_init = CR_init
        self.F_init = F_init
        self.population_size = population_size
        self.dimension = 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.local_search_factor = local_search_factor
        self.max_local_steps = max_local_steps

    def __call__(self, func):
        population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dimension)
        )
        fitness = np.array([func(ind) for ind in population])

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        F = self.F_init
        CR = self.CR_init
        evaluations = self.population_size
        local_search_steps = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c, d, e = np.random.choice(idxs, 5, replace=False)

                # Cross-mutative strategy
                mutant = (
                    population[a] + F * (population[b] - population[c]) + F * (population[d] - population[e])
                )
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                cross_points = np.random.rand(self.dimension) < CR
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                # Adaptive Local Exploration
                if np.random.rand() < self.local_search_factor:
                    local_candidate = best_solution + np.random.normal(0, 0.1, self.dimension)
                    local_candidate = np.clip(local_candidate, self.lower_bound, self.upper_bound)
                    local_fitness = func(local_candidate)
                    evaluations += 1
                    if local_fitness < best_fitness:
                        best_solution = local_candidate
                        best_fitness = local_fitness
                        local_search_steps += 1
                        if local_search_steps > self.max_local_steps:
                            local_search_steps = 0
                            self.local_search_factor /= 2  # Reduce the intensity if too many local searches

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
