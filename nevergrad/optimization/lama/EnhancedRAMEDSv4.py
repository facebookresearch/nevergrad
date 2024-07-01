import numpy as np


class EnhancedRAMEDSv4:
    def __init__(
        self,
        budget,
        population_size=50,
        crossover_rate=0.95,
        F_min=0.5,
        F_max=0.9,
        memory_size=50,
        elite_size=10,
        reinit_cycle=100,
    ):
        self.budget = budget
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.F_min = F_min
        self.F_max = F_max
        self.memory_size = memory_size
        self.elite_size = elite_size
        self.reinit_cycle = reinit_cycle
        self.lb, self.ub, self.dimension = -5.0, 5.0, 5

    def __call__(self, func):
        # Initialize population and fitness
        population = self.lb + (self.ub - self.lb) * np.random.rand(self.population_size, self.dimension)
        fitness = np.array([func(individual) for individual in population])

        # Initialize memory for good solutions and their fitness
        memory = np.empty((self.memory_size, self.dimension))
        memory_fitness = np.full(self.memory_size, np.inf)

        # Best solution tracking
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            if evaluations % self.reinit_cycle == 0 and evaluations != 0:
                # Periodic reinitialization of a portion of the population
                reinit_indices = np.random.choice(
                    range(self.population_size), size=self.population_size // 5, replace=False
                )
                population[reinit_indices] = self.lb + (self.ub - self.lb) * np.random.rand(
                    len(reinit_indices), self.dimension
                )
                fitness[reinit_indices] = np.array(
                    [func(individual) for individual in population[reinit_indices]]
                )
                evaluations += len(reinit_indices)

            for i in range(self.population_size):
                # Adaptive mutation factor with decay
                F = self.F_max - (self.F_max - self.F_min) * np.cos(2 * np.pi * evaluations / self.budget)

                # Mutation: DE/current-to-best/1
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(
                    population[i] + F * (best_solution - population[i] + a - b), self.lb, self.ub
                )

                # Crossover
                cross_points = np.random.rand(self.dimension) < self.crossover_rate
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
