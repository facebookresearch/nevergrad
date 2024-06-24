import numpy as np


class SuperOptimizedRAMEDS:
    def __init__(
        self,
        budget,
        population_size=50,
        initial_crossover_rate=0.9,
        F_min=0.5,
        F_max=1.2,
        memory_size=50,
        elite_size=10,
    ):
        self.budget = budget
        self.population_size = population_size
        self.crossover_rate = initial_crossover_rate
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
            # Dynamic adjustment of mutation factor based on the convergence rate
            dynamic_factor = evaluations / self.budget
            F = self.F_min + (self.F_max - self.F_min) * np.exp(-4 * dynamic_factor)  # Exponential decay

            # Crossover rate update based on a sinusoidal pattern
            self.crossover_rate = 0.5 + 0.45 * np.sin(np.pi * dynamic_factor)

            # Update elites from both population and memory
            combined_population = np.concatenate((population, memory[: np.argmin(memory_fitness)]))
            combined_fitness = np.concatenate((fitness, memory_fitness[: np.argmin(memory_fitness)]))
            elite_indices = np.argsort(combined_fitness)[: self.elite_size]
            elite = combined_population[elite_indices].copy()
            elite_fitness = combined_fitness[elite_indices].copy()

            for i in range(self.population_size):
                idxs = np.array([idx for idx in range(self.population_size) if idx != i])
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(
                    a + F * (b - c + elite[np.random.randint(self.elite_size)] - population[i]), lb, ub
                )

                # Crossover operation
                cross_points = np.random.rand(dimension) < self.crossover_rate
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate trial solution
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    # Memory strategy: store replaced solutions if they are better than the worst in memory
                    if trial_fitness < np.max(memory_fitness):
                        replace_idx = np.argmax(memory_fitness)
                        memory[replace_idx] = population[i]
                        memory_fitness[replace_idx] = fitness[i]

                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update the best found solution
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
