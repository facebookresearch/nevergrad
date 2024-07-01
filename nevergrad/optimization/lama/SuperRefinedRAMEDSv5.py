import numpy as np


class SuperRefinedRAMEDSv5:
    def __init__(
        self,
        budget,
        population_size=50,
        base_crossover_rate=0.9,
        F_min=0.5,
        F_max=0.9,
        memory_size=50,
        elite_size=10,
    ):
        self.budget = budget
        self.population_size = population_size
        self.base_crossover_rate = base_crossover_rate
        self.F_min = F_min
        self.F_max = F_max
        self.memory_size = memory_size
        self.elite_size = elite_size
        self.lb, self.ub, self.dimension = -5.0, 5.0, 5

    def __call__(self, func):
        # Initialize population and fitness
        population = self.lb + (self.ub - self.lb) * np.random.rand(self.population_size, self.dimension)
        fitness = np.array([func(individual) for individual in population])

        # Initialize memory and elite structures
        memory = np.empty((self.memory_size, self.dimension))
        memory_fitness = np.full(self.memory_size, np.inf)
        elite = np.empty((self.elite_size, self.dimension))
        elite_fitness = np.full(self.elite_size, np.inf)

        # Best solution tracking
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            # Update elites
            elite_indices = np.argsort(fitness)[: self.elite_size]
            elite = population[elite_indices].copy()
            elite_fitness = fitness[elite_indices].copy()

            # Adaptive mutation factor with dynamic modulation
            F = self.F_min + (self.F_max - self.F_min) * np.sin(np.pi * evaluations / self.budget)

            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                best_or_elite = elite[np.random.randint(0, self.elite_size)]
                mutant = np.clip(
                    population[i] + F * (best_or_elite - population[i] + a - b), self.lb, self.ub
                )

                # Adaptive crossover
                crossover_rate = self.base_crossover_rate * (1 - evaluations / self.budget)
                cross_points = np.random.rand(self.dimension) < crossover_rate
                trial = np.where(cross_points, mutant, population[i])

                # Selection and memory update
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
