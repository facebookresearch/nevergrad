import numpy as np


class EDMS:
    def __init__(
        self,
        budget,
        population_size=50,
        crossover_rate=0.9,
        F_base=0.7,
        F_amp=0.3,
        memory_size=40,
        elite_size=3,
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

        # Memory for good solutions, initialized with first found solutions
        memory = population[np.argsort(fitness)[: self.memory_size]].copy()
        memory_fitness = fitness[np.argsort(fitness)[: self.memory_size]].copy()

        # Elite solutions tracking
        elite = population[np.argsort(fitness)[: self.elite_size]].copy()
        elite_fitness = fitness[np.argsort(fitness)[: self.elite_size]].copy()

        # Best solution tracking
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            # Periodic elite and memory updates
            if evaluations % (self.budget // 10) == 0:
                sorted_indices = np.argsort(fitness)
                memory = population[sorted_indices[: self.memory_size]].copy()
                memory_fitness = fitness[sorted_indices[: self.memory_size]].copy()

                elite = population[sorted_indices[: self.elite_size]].copy()
                elite_fitness = fitness[sorted_indices[: self.elite_size]].copy()

            for i in range(self.population_size):
                # Adaptive mutation factor with a decaying amplitude
                F = self.F_base + self.F_amp * np.exp(-4 * evaluations / self.budget)

                # Select mutation strategy based on the evolution state
                idxs = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = population[idxs]
                if np.random.rand() < 0.3:  # Intermittently use memory in mutation
                    memory_idx = np.random.randint(0, self.memory_size)
                    mutant = np.clip(a + F * (memory[memory_idx] - b + c), lb, ub)
                else:
                    mutant = np.clip(a + F * (elite[0] - b + c), lb, ub)

                # Crossover: Binomial
                cross_points = np.random.rand(dimension) < self.crossover_rate
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial.copy()
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
