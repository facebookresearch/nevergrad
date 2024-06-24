import numpy as np


class DMDE:
    def __init__(
        self,
        budget,
        population_size=50,
        crossover_rate=0.8,
        F_base=0.5,
        F_amp=0.5,
        memory_size=50,
        elite_size=5,
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

        # Initialize population
        population = lb + (ub - lb) * np.random.rand(self.population_size, dimension)
        fitness = np.array([func(individual) for individual in population])

        # Initialize memory
        memory = np.empty((self.memory_size, dimension))
        memory_fitness = np.full(self.memory_size, np.inf)

        # Track elite solutions
        elite = population[: self.elite_size].copy()
        elite_fitness = fitness[: self.elite_size].copy()

        # Track the best solution found
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            if evaluations % (self.budget // 10) == 0:  # Update elite more frequently
                elite_indices = np.argsort(fitness)[: self.elite_size]
                elite = population[elite_indices]
                elite_fitness = fitness[elite_indices]

            for i in range(self.population_size):
                # Adaptive mutation factor with dynamic oscillation
                F = self.F_base + self.F_amp * np.sin(2 * np.pi * evaluations / self.budget)

                # Mutation: DE/current-to-best/1
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                best = (
                    best_solution if np.random.rand() < 0.75 else elite[np.random.randint(0, self.elite_size)]
                )
                mutant = np.clip(population[i] + F * (best - population[i] + b - c), lb, ub)

                # Crossover
                cross_points = np.random.rand(dimension) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimension)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    # Memorize replaced solutions
                    memory_idx = np.argmax(memory_fitness)
                    if trial_fitness < memory_fitness[memory_idx]:
                        memory[memory_idx] = population[i]
                        memory_fitness[memory_idx] = fitness[i]

                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
