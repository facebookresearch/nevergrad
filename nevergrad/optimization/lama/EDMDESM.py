import numpy as np


class EDMDESM:
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

        # Memory for good solutions
        memory = np.empty((self.memory_size, dimension))
        memory_fitness = np.full(self.memory_size, np.inf)

        # Elite solutions tracking
        elite = np.empty((self.elite_size, dimension))
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

            for i in range(self.population_size):
                # Adaptive mutation factor
                F = self.F_base + self.F_amp * np.sin(2 * np.pi * evaluations / self.budget)

                # Mutation using best, random from population and elite
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                best_or_elite = (
                    best_solution if np.random.rand() < 0.85 else elite[np.random.randint(0, self.elite_size)]
                )
                mutant = np.clip(a + F * (best_or_elite - a + b - c), lb, ub)

                # Crossover: Binomial with adaptive technique
                cross_points = np.random.rand(dimension) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimension)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    # Update memory by replacing the worst entry
                    worst_idx = np.argmax(memory_fitness)
                    if fitness[i] < memory_fitness[worst_idx]:
                        memory[worst_idx] = population[i]
                        memory_fitness[worst_idx] = fitness[i]

                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
