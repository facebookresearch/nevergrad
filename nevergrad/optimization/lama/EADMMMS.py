import numpy as np


class EADMMMS:
    def __init__(
        self,
        budget,
        population_size=100,
        crossover_rate=0.95,
        F_base=0.5,
        F_amp=0.5,
        memory_size=50,
        elite_size=10,
    ):
        self.budget = budget
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.F_base = F_base
        self.F_amp = F_amp
        self.memory_size = memory_size
        self.elite_size = elite_size

    def __call__(self, func):
        lb, ub, dimension = -5.0, 5.0, 5

        # Initialize population uniformly within bounds
        population = lb + (ub - lb) * np.random.rand(self.population_size, dimension)
        fitness = np.array([func(individual) for individual in population])

        # Memory for good solutions and elite solutions
        memory = np.empty((self.memory_size, dimension))
        memory_fitness = np.full(self.memory_size, np.inf)
        elite = np.empty((self.elite_size, dimension))
        elite_fitness = np.full(self.elite_size, np.inf)

        # Best solution tracking
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            # Update elite and memory periodically
            if evaluations % (self.budget // 20) == 0:
                sorted_indices = np.argsort(fitness)
                elite[:] = population[sorted_indices[: self.elite_size]]
                elite_fitness[:] = fitness[sorted_indices[: self.elite_size]]
                memory[:] = population[sorted_indices[: self.memory_size]]
                memory_fitness[:] = fitness[sorted_indices[: self.memory_size]]

            for i in range(self.population_size):
                # Adaptive mutation factor using wave pattern
                F = self.F_base + self.F_amp * np.sin(2 * np.pi * evaluations / self.budget)

                # Mutation strategy incorporating best, memory, and elite solutions
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                best_or_elite = (
                    best_solution if np.random.rand() < 0.85 else elite[np.random.randint(self.elite_size)]
                )
                memory_contrib = memory[np.random.randint(self.memory_size)]
                mutant = np.clip(a + F * (best_or_elite - a + memory_contrib - b), lb, ub)

                # Crossover using binomial method
                cross_points = np.random.rand(dimension) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimension)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
