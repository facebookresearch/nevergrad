import numpy as np


class HyperRAMEDS:
    def __init__(
        self,
        budget,
        population_size=50,
        crossover_rate=0.9,
        F_min=0.4,
        F_max=0.8,
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
        self.lb, self.ub, self.dimension = -5.0, 5.0, 5

    def __call__(self, func):
        # Initialize population and fitness
        population = self.lb + (self.ub - self.lb) * np.random.rand(self.population_size, self.dimension)
        fitness = np.array([func(individual) for individual in population])

        # Initialize memory for good solutions and their fitness
        memory = np.empty((self.memory_size, self.dimension))
        memory_fitness = np.full(self.memory_size, np.inf)

        # Initialize elite solutions and their fitness
        elite = np.empty((self.elite_size, self.dimension))
        elite_fitness = np.full(self.elite_size, np.inf)

        # Best solution tracking
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            # Update elites based on fitness
            elite_indices = np.argsort(fitness)[: self.elite_size]
            elite = population[elite_indices].copy()
            elite_fitness = fitness[elite_indices].copy()

            for i in range(self.population_size):
                # Adaptive mutation factor with non-linear modulation
                F = self.F_max - (self.F_max - self.F_min) * (1 - np.exp(-4 * evaluations / self.budget))

                # Mutation: DE/best/1/binomial with adaptive mutation based on elite and memory
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b = population[np.random.choice(idxs, 2, replace=False)]
                best_or_elite = (
                    best_solution if np.random.rand() < 0.65 else elite[np.random.randint(0, self.elite_size)]
                )
                mutant = np.clip(best_or_elite + F * (a - b), self.lb, self.ub)

                # Crossover
                cross_points = np.random.rand(self.dimension) < self.crossover_rate
                trial = np.where(cross_points, mutant, population[i])

                # Evaluation and selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update memory if the trial solution is better than the worst in memory
                    worst_memory_idx = np.argmax(memory_fitness)
                    if trial_fitness < memory_fitness[worst_memory_idx]:
                        memory[worst_memory_idx] = trial.copy()
                        memory_fitness[worst_memory_idx] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
