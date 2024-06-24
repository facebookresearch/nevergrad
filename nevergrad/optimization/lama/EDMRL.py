import numpy as np


class EDMRL:
    def __init__(
        self,
        budget,
        population_size=50,
        crossover_rate=0.9,
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

        # Initialize population within bounds
        population = lb + (ub - lb) * np.random.rand(self.population_size, dimension)
        fitness = np.array([func(individual) for individual in population])

        # Memory for good solutions
        memory = np.empty((self.memory_size, dimension))
        memory_fitness = np.full(self.memory_size, np.inf)

        # Track elite solutions
        elite = np.empty((self.elite_size, dimension))
        elite_fitness = np.full(self.elite_size, np.inf)

        # Track the best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            # Update elite solutions more frequently
            if evaluations % (self.budget // 25) == 0:
                elite_update_indices = np.argsort(fitness)[: self.elite_size]
                elite = population[elite_update_indices].copy()
                elite_fitness = fitness[elite_update_indices].copy()

            for i in range(self.population_size):
                # Adaptive mutation factor with dynamic amplitude adjustment
                F = self.F_base + self.F_amp * np.cos(2 * np.pi * evaluations / self.budget)

                # Mutation strategy, incorporating reflective learning from memory
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                # Choose between the best solution and a memory solution for mutation base
                memory_used = False
                if np.random.rand() < 0.1 and np.any(memory_fitness < np.inf):
                    memory_idx = np.argmin(memory_fitness)
                    base_solution = memory[memory_idx]
                    memory_used = True
                else:
                    base_solution = (
                        best_solution
                        if np.random.rand() < 0.85
                        else elite[np.random.randint(0, self.elite_size)]
                    )

                mutant = np.clip(base_solution + F * (b - c), lb, ub)

                # Binomial crossover with adaptive mutation incorporation
                cross_points = np.random.rand(dimension) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimension)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Replace worst memory entry with current if better and was used
                    if memory_used:
                        worst_memory_idx = np.argmax(memory_fitness)
                        memory[worst_memory_idx] = trial
                        memory_fitness[worst_memory_idx] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
