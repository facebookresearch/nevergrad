import numpy as np


class HyperEvolvedRAMEDS:
    def __init__(
        self,
        budget,
        population_size=50,
        initial_crossover_rate=0.9,
        F_min=0.8,
        F_max=1.2,
        memory_size=100,
        elite_size=5,
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
            # Dynamic adjustment of mutation factor based on sigmoid of diversity
            diversity = np.std(population, axis=0).mean()
            F = self.F_min + (self.F_max - self.F_min) * np.exp(-diversity)

            # Dynamic crossover rate adjustment
            self.crossover_rate = 0.5 + 0.45 * np.sin(np.pi * evaluations / self.budget)

            # Update elites
            elite_indices = np.argsort(fitness)[: self.elite_size]
            elite = population[elite_indices].copy()
            elite_fitness = fitness[elite_indices].copy()

            for i in range(self.population_size):
                idxs = np.array([idx for idx in range(self.population_size) if idx != i])
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), lb, ub)

                # Dynamic crossover
                cross_points = np.random.rand(dimension) < self.crossover_rate
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate trial solution
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    # Adaptive memory strategy
                    if trial_fitness < np.max(memory_fitness):
                        replace_idx = np.argmax(memory_fitness)
                        memory[replace_idx] = population[i]  # Store replaced solution in memory
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
