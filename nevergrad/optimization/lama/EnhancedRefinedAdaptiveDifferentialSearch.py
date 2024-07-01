import numpy as np


class EnhancedRefinedAdaptiveDifferentialSearch:
    def __init__(
        self,
        budget,
        population_size=50,
        crossover_rate=0.8,
        F_min=0.4,
        F_max=1.0,
        memory_size=50,
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

        # Initialize memory for good solutions and their fitness
        memory = np.empty((self.memory_size, dimension))
        memory_fitness = np.full(self.memory_size, np.inf)

        # Initialize elite solutions and their fitness
        elite = np.empty((self.elite_size, dimension))
        elite_fitness = np.full(self.elite_size, np.inf)

        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            # Update elites
            elite_indices = np.argsort(fitness)[: self.elite_size]
            elite = population[elite_indices].copy()
            elite_fitness = fitness[elite_indices].copy()

            for i in range(self.population_size):
                F = self.F_max - (self.F_max - self.F_min) * np.exp(-4 * evaluations / self.budget)

                # Mutation: DE/rand/1
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), lb, ub)

                # Crossover
                cross_points = np.random.rand(dimension) < self.crossover_rate
                trial = np.where(cross_points, mutant, population[i])

                # Selection and updating memory
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    worst_idx = np.argmax(memory_fitness)
                    if memory_fitness[worst_idx] > fitness[i]:
                        memory[worst_idx] = population[i].copy()
                        memory_fitness[worst_idx] = fitness[i]

                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
