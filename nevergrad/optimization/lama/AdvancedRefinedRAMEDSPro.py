import numpy as np


class AdvancedRefinedRAMEDSPro:
    def __init__(
        self,
        budget,
        population_size=50,
        crossover_rate=0.9,
        F_min=0.5,
        F_max=1.0,
        memory_size=20,
        elite_size=5,
    ):
        self.budget = budget
        self.population_size = population_size
        self.crossover_rate_initial = crossover_rate
        self.F_min = F_min
        self.F_max = F_max
        self.memory_size = memory_size
        self.elite_size = elite_size

    def __call__(self, func):
        lb, ub, dimension = -5.0, 5.0, 5

        # Initialize population and fitness
        population = lb + (ub - lb) * np.random.rand(self.population_size, dimension)
        fitness = np.array([func(individual) for individual in population])

        # Initialize memory with best initial individuals
        memory = np.empty((self.memory_size, dimension))
        memory_fitness = np.full(self.memory_size, np.inf)

        # Best solution tracking
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            # Update crossover rate
            crossover_rate = self.crossover_rate_initial * (1 - evaluations / self.budget) + 0.1

            for i in range(self.population_size):
                # Adaptive mutation factor with feedback modulation
                F = self.F_max - (self.F_max - self.F_min) * (evaluations / self.budget)

                # Mutation: Hybrid strategy using best, random, and worst
                indices = np.random.choice(self.population_size, 3, replace=False)
                r1, r2, r3 = population[indices]
                random_worst_idx = np.argmax(fitness)
                random_best_or_worst = (
                    best_solution if np.random.rand() < 0.5 else population[random_worst_idx]
                )
                mutant = np.clip(r1 + F * (random_best_or_worst - r2 + r3 - population[i]), lb, ub)

                # Crossover
                cross_points = np.random.rand(dimension) < crossover_rate
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

                # Update memory if necessary
                worst_memory_idx = np.argmax(memory_fitness)
                if fitness[i] < memory_fitness[worst_memory_idx]:
                    memory[worst_memory_idx] = population[i].copy()
                    memory_fitness[worst_memory_idx] = fitness[i]

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
