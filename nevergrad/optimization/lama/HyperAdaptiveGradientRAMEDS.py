import numpy as np


class HyperAdaptiveGradientRAMEDS:
    def __init__(
        self,
        budget,
        population_size=50,
        init_crossover=0.8,
        F_min=0.4,
        F_max=0.9,
        memory_size=100,
        elite_size=5,
    ):
        self.budget = budget
        self.population_size = population_size
        self.crossover_rate = init_crossover
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
            progress = evaluations / self.budget
            # Adaptive mutation factor
            F = self.F_min + (self.F_max - self.F_min) * np.sin(np.pi * progress)

            # Adaptive crossover based on population fitness variance
            fitness_variance = np.var(fitness)
            self.crossover_rate = self.crossover_rate * (1 - np.exp(-fitness_variance))

            # Update elites
            elite_indices = np.argsort(fitness)[: self.elite_size]
            elite, elite_fitness = population[elite_indices].copy(), fitness[elite_indices].copy()

            for i in range(self.population_size):
                idxs = np.array([idx for idx in range(self.population_size) if idx != i])
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), lb, ub)

                # Crossover
                cross_points = np.random.rand(dimension) < self.crossover_rate
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate trial solution
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    # Memory update focusing on the worst replaced by better
                    if trial_fitness < np.max(memory_fitness):
                        worst_memory_idx = np.argmax(memory_fitness)
                        memory[worst_memory_idx] = trial
                        memory_fitness[worst_memory_idx] = trial_fitness

                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update the best found solution
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
