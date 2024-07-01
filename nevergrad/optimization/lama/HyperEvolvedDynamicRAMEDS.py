import numpy as np


class HyperEvolvedDynamicRAMEDS:
    def __init__(
        self, budget, population_size=50, crossover_base=0.7, F0=0.5, F1=0.9, memory_size=50, elite_size=10
    ):
        self.budget = budget
        self.population_size = population_size
        self.crossover_base = crossover_base  # Base rate for crossover
        self.F0 = F0  # Base mutation factor
        self.F1 = F1  # Maximum mutation factor
        self.memory_size = memory_size
        self.elite_size = elite_size

    def __call__(self, func):
        lb, ub, dimension = -5.0, 5.0, 5

        # Initialize population and fitness
        population = lb + (ub - lb) * np.random.rand(self.population_size, dimension)
        fitness = np.array([func(individual) for individual in population])

        # Memory and elite initialization
        memory = np.empty((self.memory_size, dimension))
        memory_fitness = np.full(self.memory_size, np.inf)
        elite = np.empty((self.elite_size, dimension))
        elite_fitness = np.full(self.elite_size, np.inf)

        # Best solution tracking
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            # Dynamically adjust mutation factor using a logistic growth model
            F_adjustment = (self.F1 - self.F0) * evaluations / self.budget
            F = self.F0 + F_adjustment * np.random.rand()

            # Update elites periodically
            if evaluations % (self.budget // 10) == 0:
                elite_indices = np.argsort(fitness)[: self.elite_size]
                elite = population[elite_indices].copy()
                elite_fitness = fitness[elite_indices].copy()

            for i in range(self.population_size):
                idxs = np.array([idx for idx in range(self.population_size) if idx != i])
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(best_solution + F * (b - c), lb, ub)

                # Dynamic crossover rate adjusted by improvements in the fitness
                improvements = np.max(fitness) - fitness
                dynamic_cr = self.crossover_base + 0.25 * (improvements[i] / (np.max(improvements) + 1e-10))
                cross_points = np.random.rand(dimension) < dynamic_cr
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate trial solution
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Memory update with better solutions
                    if trial_fitness < np.max(memory_fitness):
                        worst_memory_idx = np.argmax(memory_fitness)
                        memory[worst_memory_idx] = trial
                        memory_fitness[worst_memory_idx] = trial_fitness

                    # Update the best found solution
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
