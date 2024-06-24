import numpy as np


class RefinedOptimalEnhancedRAMEDS:
    def __init__(
        self,
        budget,
        population_size=50,
        crossover_rate=0.93,
        F_min=0.45,
        F_max=0.85,
        memory_size=50,
        elite_size=10,
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

        # Best solution tracking
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            # Periodically shuffle elites to prevent local minima stagnation
            if evaluations % (self.budget // 10) == 0:
                np.random.shuffle(elite)

            for i in range(self.population_size):
                # Adaptive mutation factor with an improved modulation strategy
                F = self.F_max - (self.F_max - self.F_min) * np.sin(2 * np.pi * evaluations / self.budget)

                # Mutation: DE/current-to-best/1 with improved selection for mutation basis
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                best_or_elite = (
                    best_solution if np.random.rand() < 0.65 else elite[np.random.randint(0, self.elite_size)]
                )
                mutant = np.clip(population[i] + F * (best_or_elite - population[i] + a - b), lb, ub)

                # Crossover
                cross_points = np.random.rand(dimension) < self.crossover_rate
                trial = np.where(cross_points, mutant, population[i])

                # Selection and updating memory
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    # Update memory strategically
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
