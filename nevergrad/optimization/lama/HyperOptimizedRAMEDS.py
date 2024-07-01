import numpy as np


class HyperOptimizedRAMEDS:
    def __init__(
        self,
        budget,
        population_size=50,
        crossover_rate=0.95,
        F_min=0.2,
        F_max=0.8,
        memory_size=50,
        elite_size=5,
    ):
        self.budget = budget
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.F_min = F_min  # Lower starting mutation factor for finer adjustments
        self.F_max = F_max  # Lower max mutation for less disruptive mutations
        self.memory_size = memory_size
        self.elite_size = elite_size  # Smaller elite group for more focused exploitation

    def __call__(self, func):
        lb, ub, dimension = -5.0, 5.0, 5

        # Initialize population and fitness
        population = lb + (ub - lb) * np.random.rand(self.population_size, dimension)
        fitness = np.array([func(individual) for individual in population])

        # Initialize memory for good solutions and their fitness
        memory = np.empty((self.memory_size, dimension))
        memory_fitness = np.full(self.memory_size, np.inf)

        # Best solution tracking
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            # Dynamically adjust mutation factor based on progress and performance variance
            progress = evaluations / self.budget
            F = self.F_min + (self.F_max - self.F_min) * np.tanh(4 * (progress - 0.5))

            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), lb, ub)

                # Adaptive Crossover: Consider progress to decide on crossover rate
                adaptive_cr = self.crossover_rate * (1 - progress) + 0.5 * progress
                cross_points = np.random.rand(dimension) < adaptive_cr
                trial = np.where(cross_points, mutant, population[i])

                # Evaluation and selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update memory strategically
                    if trial_fitness < np.max(memory_fitness):
                        replace_idx = np.argmax(memory_fitness)
                        memory[replace_idx] = trial
                        memory_fitness[replace_idx] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
