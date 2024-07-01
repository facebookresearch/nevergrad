import numpy as np


class EHDAM:
    def __init__(
        self,
        budget,
        population_size=80,
        crossover_rate=0.8,
        F_base=0.5,
        F_amp=0.5,
        memory_size=150,
        elite_size=20,
    ):
        self.budget = budget
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.F_base = F_base
        self.F_amp = F_amp
        self.memory_size = memory_size
        self.elite_size = elite_size

    def __call__(self, func):
        lb = -5.0
        ub = 5.0
        dimension = 5

        # Initialize population uniformly within bounds
        population = lb + (ub - lb) * np.random.rand(self.population_size, dimension)
        fitness = np.array([func(individual) for individual in population])

        # Memory for good solutions
        memory = np.empty((0, dimension))
        # Elite solutions tracking
        elite = np.empty((self.elite_size, dimension))

        # Best solution tracking
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            # Update elite solutions periodically
            if evaluations % (self.budget // 30) == 0:
                elite_idx = np.argsort(fitness)[: self.elite_size]
                elite = population[elite_idx]

            for i in range(self.population_size):
                # Adaptive mutation factor that changes dynamically
                F = self.F_base + self.F_amp * np.sin(2 * np.pi * evaluations / self.budget)

                # Mutation: DE/rand-to-best-with-memory/1 strategy
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mem_contrib = (
                    memory[np.random.randint(0, np.clip(memory.shape[0], 1, self.memory_size))]
                    if memory.size > 0
                    else 0
                )
                best = (
                    best_solution if np.random.rand() < 0.75 else elite[np.random.randint(0, self.elite_size)]
                )
                mutant = np.clip(a + F * (best - a + b - c + mem_contrib), lb, ub)

                # Crossover
                cross_points = np.random.rand(dimension) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimension)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    # Update memory with the old good solutions more aggressively
                    if memory.shape[0] < self.memory_size:
                        memory = np.vstack([memory, population[i]])
                    elif np.random.rand() < 0.3:  # More frequent replacement of memory entries
                        memory[np.random.randint(0, self.memory_size)] = population[i]

                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
