import numpy as np


class EnhancedRAMEDSProV2:
    def __init__(
        self,
        budget,
        population_size=50,
        crossover_rate_initial=0.95,
        F_min=0.1,
        F_max=0.9,
        memory_size=50,
        elite_size=10,
    ):
        self.budget = budget
        self.population_size = population_size
        self.crossover_rate_initial = crossover_rate_initial
        self.F_min = F_min
        self.F_max = F_max
        self.memory_size = memory_size
        self.elite_size = elite_size

    def __call__(self, func):
        lb, ub, dimension = -5.0, 5.0, 5

        # Initialize population and fitness
        population = lb + (ub - lb) * np.random.rand(self.population_size, dimension)
        fitness = np.array([func(individual) for individual in population])

        # Memory and elite initialization
        memory = population[np.argsort(fitness)[: self.memory_size]].copy()
        memory_fitness = fitness[np.argsort(fitness)[: self.memory_size]].copy()
        elite = population[np.argsort(fitness)[: self.elite_size]].copy()
        elite_fitness = fitness[np.argsort(fitness)[: self.elite_size]].copy()

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            # Adaptive mutation and crossover rates
            progress = evaluations / self.budget
            F = self.F_max - (self.F_max - self.F_min) * np.sin(np.pi * progress)
            crossover_rate = self.crossover_rate_initial * (1 - progress) + 0.1 * progress

            for i in range(self.population_size):
                # Mutation with elite and memory consideration
                elites_indices = np.random.choice(self.elite_size, 2, replace=False)
                memory_indices = np.random.choice(self.memory_size, 2, replace=False)
                x1, x2 = elite[elites_indices]
                m1, m2 = memory[memory_indices]

                mutant = np.clip(population[i] + F * (x1 - x2 + m1 - m2), lb, ub)

                # Crossover
                cross_points = np.random.rand(dimension) < crossover_rate
                trial = np.where(cross_points, mutant, population[i])

                # Selection and memory update
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                    # Memory update
                    if trial_fitness < np.max(memory_fitness):
                        worst_idx = np.argmax(memory_fitness)
                        memory[worst_idx] = trial
                        memory_fitness[worst_idx] = trial_fitness

                    # Elite update
                    if trial_fitness < np.max(elite_fitness):
                        worst_elite_idx = np.argmax(elite_fitness)
                        elite[worst_elite_idx] = trial
                        elite_fitness[worst_elite_idx] = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
