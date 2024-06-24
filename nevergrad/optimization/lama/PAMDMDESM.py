import numpy as np


class PAMDMDESM:
    def __init__(
        self,
        budget,
        population_size=50,
        crossover_rate=0.95,
        F_base=0.5,
        F_amp=0.5,
        memory_size=50,
        elite_size=5,
    ):
        self.budget = budget
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.F_base = F_base
        self.F_amp = F_amp
        self.memory_size = memory_size
        self.elite_size = elite_size

    def __call__(self, func):
        lb, ub, dimension = -5.0, 5.0, 5

        # Initialize population uniformly within bounds
        population = lb + (ub - lb) * np.random.rand(self.population_size, dimension)
        fitness = np.array([func(individual) for individual in population])

        # Memory for good solutions, initialized empty and filled as better solutions are found
        memory = np.zeros((self.memory_size, dimension))
        memory_fitness = np.full(self.memory_size, np.inf)

        # Elite solutions tracking
        elite = np.zeros((self.elite_size, dimension))
        elite_fitness = np.full(self.elite_size, np.inf)

        # Best solution tracking
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            # Update elite solutions periodically
            if evaluations % (self.budget // 20) == 0:
                elite_idx = np.argsort(fitness)[: self.elite_size]
                elite = population[elite_idx].copy()
                elite_fitness = fitness[elite_idx].copy()

            # Memory incorporation in crossover and mutation
            if evaluations % (self.budget // 10) == 0 and np.any(memory_fitness < np.inf):
                memory_selection = memory[np.argmin(memory_fitness)]
            else:
                memory_selection = None

            for i in range(self.population_size):
                # Adaptive mutation factor
                F = self.F_base + self.F_amp * np.random.normal()

                # Select mutation strategy based on progression of evaluations
                idxs = np.random.choice(
                    [idx for idx in range(self.population_size) if idx != i], 3, replace=False
                )
                a, b, c = population[idxs]
                if memory_selection is not None and np.random.rand() < 0.2:
                    mutant = np.clip(a + F * (memory_selection - b + c), lb, ub)
                else:
                    mutant = np.clip(a + F * (best_solution - b + c), lb, ub)

                # Crossover: Uniform with adaptive probability
                cross_points = np.random.rand(dimension) < self.crossover_rate
                trial = np.where(cross_points, mutant, population[i])

                # Evaluation and selection
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    # Update memory with current solution before it is replaced
                    if fitness[i] < np.max(memory_fitness):
                        worst_mem_idx = np.argmax(memory_fitness)
                        memory[worst_mem_idx] = population[i].copy()
                        memory_fitness[worst_mem_idx] = fitness[i]

                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial.copy()
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
