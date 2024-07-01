import numpy as np


class AADMEM:
    def __init__(
        self, budget, population_size=50, crossover_rate=0.9, F_base=0.5, F_amp=0.3, memory_size=100
    ):
        self.budget = budget
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.F_base = F_base
        self.F_amp = F_amp
        self.memory_size = memory_size

    def __call__(self, func):
        lb = -5.0
        ub = 5.0
        dimension = 5

        # Initialize population uniformly within bounds
        population = lb + (ub - lb) * np.random.rand(self.population_size, dimension)
        fitness = np.array([func(individual) for individual in population])

        # Memory to store good solutions
        memory = np.empty((0, dimension))

        # Best solution tracking
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Adaptive mutation factor that changes over time to increase exploration
                F = self.F_base + self.F_amp * np.sin(np.pi * evaluations / self.budget)

                # Mutation vectors from population and occasionally from memory
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                if memory.shape[0] > 0 and np.random.rand() < 0.1:  # 10% chance to use memory
                    a = memory[np.random.randint(0, memory.shape[0])]

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
                    # Update memory with the old solution
                    if memory.shape[0] < self.memory_size:
                        memory = np.vstack([memory, population[i]])
                    else:
                        # Replace a random entry in memory
                        memory[np.random.randint(0, self.memory_size)] = population[i]

                    # Update population with the new better solution
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update the best solution
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                # Check if budget exhausted
                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
