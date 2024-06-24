import numpy as np


class AdvancedRefinedAdaptiveMemoryEnhancedSearch:
    def __init__(
        self,
        budget,
        population_size=50,
        crossover_base=0.7,
        F_min=0.5,
        F_max=1,
        memory_size=50,
        elite_size=10,
        aging_factor=0.9,
    ):
        self.budget = budget
        self.population_size = population_size
        self.crossover_base = crossover_base  # Base rate for crossover, adaptively adjusted
        self.F_min = F_min
        self.F_max = F_max
        self.memory_size = memory_size
        self.elite_size = elite_size
        self.aging_factor = aging_factor  # Controls how fast memory ages

    def __call__(self, func):
        lb, ub, dimension = -5.0, 5.0, 5

        # Initialize population and fitness
        population = lb + (ub - lb) * np.random.rand(self.population_size, dimension)
        fitness = np.array([func(individual) for individual in population])

        # Initialize memory and elite structures
        memory = np.empty((self.memory_size, dimension))
        memory_fitness = np.full(self.memory_size, np.inf)
        elite = np.empty((self.elite_size, dimension))
        elite_fitness = np.full(self.elite_size, np.inf)

        # Track the best solution and its fitness
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            # Update elite solutions
            elite_indices = np.argsort(fitness)[: self.elite_size]
            elite = population[elite_indices].copy()
            elite_fitness = fitness[elite_indices].copy()

            for i in range(self.population_size):
                # Dynamic mutation factor incorporating feedback from memory
                F = self.F_max - (self.F_max - self.F_min) * np.sin(np.pi * evaluations / self.budget)

                # Select mutation strategy: DE/current-to-best/1 or DE/rand/1 based on fitness
                if np.random.rand() < 0.5:
                    idxs = [idx for idx in range(self.population_size) if idx != i]
                    a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(population[i] + F * (best_solution - population[i] + a - b), lb, ub)
                else:
                    a, b = population[np.random.choice(self.population_size, 2, replace=False)]
                    mutant = np.clip(a + F * (b - a), lb, ub)

                # Adaptive crossover rate based on individual performance
                crossover_rate = self.crossover_base + (1 - self.crossover_base) * (
                    fitness[i] / np.max(fitness)
                )
                cross_points = np.random.rand(dimension) < crossover_rate
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    # Memory aging and update
                    aged_fitness = memory_fitness * self.aging_factor
                    worst_idx = np.argmax(aged_fitness)
                    if aged_fitness[worst_idx] > fitness[i]:
                        memory[worst_idx] = population[i].copy()
                        memory_fitness[worst_idx] = fitness[i]

                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update the best solution found
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
