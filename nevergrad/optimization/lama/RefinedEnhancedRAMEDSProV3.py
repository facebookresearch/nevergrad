import numpy as np


class RefinedEnhancedRAMEDSProV3:
    def __init__(
        self,
        budget,
        population_size=100,
        crossover_rate=0.9,
        F_min=0.5,
        F_max=0.9,
        memory_size=30,
        elite_size=5,
    ):
        self.budget = budget
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.F_min = F_min
        self.F_max = F_max
        self.memory_size = memory_size
        self.elite_size = elite_size

    def __call__(self, func):
        lb, ub, dimension = -5.0, 5.0, 5

        # Initialize population and fitness
        population = lb + (ub - lb) * np.random.rand(self.population_size, dimension)
        fitness = np.array([func(individual) for individual in population])

        # Initialize memory and elite storage
        memory = population[: self.memory_size].copy()
        memory_fitness = fitness[: self.memory_size].copy()
        elite = population[: self.elite_size].copy()
        elite_fitness = fitness[: self.elite_size].copy()

        # Best solution tracking
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            # Update elite and memory based on current best
            for i in range(self.population_size):
                # Mutation: DE/rand-to-best/1 with elite adaptation
                F = self.F_max - (self.F_max - self.F_min) * np.cos(np.pi * evaluations / self.budget)
                a, b, c = population[np.random.choice(self.population_size, 3, replace=False)]
                mutant = np.clip(a + F * (best_solution - b + c), lb, ub)

                # Crossover
                cross_points = np.random.rand(dimension) < self.crossover_rate
                trial = np.where(cross_points, mutant, population[i])

                # Evaluation and selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update memory and elite
                    if trial_fitness < np.max(memory_fitness):
                        worst_mem_idx = np.argmax(memory_fitness)
                        memory[worst_mem_idx] = trial
                        memory_fitness[worst_mem_idx] = trial_fitness

                    if trial_fitness < np.max(elite_fitness):
                        worst_elite_idx = np.argmax(elite_fitness)
                        elite[worst_elite_idx] = trial
                        elite_fitness[worst_elite_idx] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
