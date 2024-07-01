import numpy as np


class RefinedRAMEDSPro:
    def __init__(
        self,
        budget,
        population_size=50,
        crossover_rate=0.8,
        F_min=0.4,
        F_max=1.0,
        memory_size=20,
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

        # Initialize memory with best initial individuals
        memory_indices = np.argsort(fitness)[: self.memory_size]
        memory = population[memory_indices].copy()
        memory_fitness = fitness[memory_indices].copy()

        # Best solution tracking
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Adaptive mutation factor with successful feedback modulation
                F = np.interp(evaluations, [0, self.budget], [self.F_max, self.F_min])
                if evaluations % 20 == 0 and i == 0:
                    F += (0.5 - np.random.rand()) * 0.2  # Small random deviation every 20 evaluations

                # Mutation: DE/rand-to-best/1
                indices = np.random.choice(self.population_size, 2, replace=False)
                r1, r2 = population[indices]
                mutant = np.clip(best_solution + F * (r1 - r2), lb, ub)

                # Crossover
                cross_points = np.random.rand(dimension) < self.crossover_rate
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
                max_memory_f_idx = np.argmax(memory_fitness)
                if fitness[i] < memory_fitness[max_memory_f_idx]:
                    memory[max_memory_f_idx] = population[i]
                    memory_fitness[max_memory_f_idx] = fitness[i]

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
