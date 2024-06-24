import numpy as np


class RefinedHyperEvolvedDynamicRAMEDS:
    def __init__(
        self,
        budget,
        population_size=50,
        crossover_base=0.8,
        F_min=0.5,
        F_max=0.9,
        memory_size=50,
        elite_size=10,
    ):
        self.budget = budget
        self.population_size = population_size
        self.crossover_base = crossover_base  # Base rate for crossover
        self.F_min = F_min  # Minimum mutation factor
        self.F_max = F_max  # Maximum mutation factor
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

        # Tracking best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            # Dynamically adjust mutation factor based on evaluations and variance of fitness
            fitness_variance = np.var(fitness)
            F = self.F_min + (self.F_max - self.F_min) * (1 - np.exp(-fitness_variance)) * np.random.rand()

            if evaluations % (self.budget // 10) == 0:
                # Periodically update elite repository
                sorted_indices = np.argsort(fitness)
                elite = population[sorted_indices[: self.elite_size]].copy()
                elite_fitness = fitness[sorted_indices[: self.elite_size]].copy()

            for i in range(self.population_size):
                idxs = np.array([idx for idx in range(self.population_size) if idx != i])
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(best_solution + F * (b - c), lb, ub)

                # Adaptive crossover rate based on the fitness improvement rate
                improv_rate = np.abs(fitness[i] - np.mean(fitness)) / (np.std(fitness) + 1e-8)
                cross_rate = self.crossover_base + 0.2 * np.tanh(improv_rate)
                cross_points = np.random.rand(dimension) < cross_rate
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
