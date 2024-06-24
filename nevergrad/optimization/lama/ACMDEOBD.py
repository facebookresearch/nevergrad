import numpy as np


class ACMDEOBD:
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

    def opposition_based_learning(self, population):
        return self.lower_bound + self.upper_bound - population

    def __call__(self, func):
        # Initialize population with Opposition-Based Learning
        initial_population = np.random.uniform(
            self.lower_bound, self.upper_bound, (self.population_size, self.dimension)
        )
        opposite_population = self.opposition_based_learning(initial_population)
        combined_population = np.vstack((initial_population, opposite_population))
        fitness = np.array([func(ind) for ind in combined_population])
        indices = np.argsort(fitness)[: self.population_size]
        population = combined_population[indices]
        fitness = fitness[indices]

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        F = self.F_init
        CR = self.CR_init
        evaluations = 2 * self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c, d, e = np.random.choice(idxs, 5, replace=False)

                # Adaptive parameters
                F = self.F_init * (1 - evaluations / self.budget)
                CR = self.CR_init * (evaluations / self.budget)

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

            # Periodic Opposition-Based Learning
            if evaluations % (self.population_size * 5) == 0:
                population = self.opposition_based_learning(population)
                fitness = np.array([func(ind) for ind in population])

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
