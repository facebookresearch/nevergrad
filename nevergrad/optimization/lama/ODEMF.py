import numpy as np


class ODEMF:
    def __init__(
        self,
        budget,
        population_size=50,
        crossover_rate=0.85,
        F_min=0.5,
        F_max=0.9,
        memory_size=100,
        elite_size=10,
    ):
        self.budget = budget
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.F_min = F_min
        self.F_max = F_max
        self.memory_size = memory_size
        self.elite_size = elite_size

    def __call__(self, func):
        lb = -5.0
        ub = 5.0
        dimension = 5

        # Initialize population uniformly within the bounds
        population = lb + (ub - lb) * np.random.rand(self.population_size, dimension)
        fitness = np.array([func(individual) for individual in population])

        # Memory for storing good solutions
        memory = np.empty((0, dimension))
        # Tracking elite solutions
        elite = np.empty((self.elite_size, dimension))

        # Best solution tracking
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            # Update elite solutions periodically
            if evaluations % (self.budget // 10) == 0:
                elite_idx = np.argsort(fitness)[: self.elite_size]
                elite = population[elite_idx]

            for i in range(self.population_size):
                # Adaptive mutation factor based on feedback mechanism
                F = self.F_min + (self.F_max - self.F_min) * evaluations / self.budget * np.exp(
                    -4 * (best_fitness - fitness[i]) ** 2
                )

                # Mutation: Differential Evolution strategy with feedback on fitness
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), lb, ub)

                # Crossover
                cross_points = np.random.rand(dimension) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimension)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    # Update memory with the old good solutions
                    if memory.shape[0] < self.memory_size:
                        memory = np.vstack([memory, population[i]])
                    elif np.random.rand() < 0.2:  # More frequent replacement in memory
                        memory[np.random.randint(0, self.memory_size)] = population[i]

                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
