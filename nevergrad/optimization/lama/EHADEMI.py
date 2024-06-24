import numpy as np


class EHADEMI:
    def __init__(
        self,
        budget,
        population_size=50,
        crossover_rate=0.92,
        F_base=0.58,
        F_amp=0.42,
        memory_size=120,
        elite_size=12,
    ):
        self.budget = budget
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.F_base = F_base
        self.F_amp = F_amp
        self.memory_size = memory_size
        self.elite_size = elite_size

    def __call__(self, func):
        lb = -5.0
        ub = 5.0
        dimension = 5

        # Initialize population uniformly within the bounds
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
                elite_indices = np.argsort(fitness)[: self.elite_size]
                elite = population[elite_indices].copy()
                elite_fitness = fitness[elite_indices].copy()

            for i in range(self.population_size):
                # Adaptive mutation factor that changes dynamically
                F = self.F_base + self.F_amp * np.sin(2 * np.pi * evaluations / self.budget)

                # Mutation: DE/rand-to-best/1/b strategy
                idxs = [idx for idx in range(self.population_size) if idx != i]
                random_indices = np.random.choice(idxs, 3, replace=False)
                a, b, c = population[random_indices]

                best_or_elite = (
                    best_solution if np.random.rand() < 0.8 else elite[np.random.randint(0, self.elite_size)]
                )
                mutant = np.clip(a + F * (best_or_elite - a + b - c), lb, ub)

                # Crossover: Binomial with adaptive rate
                cross_points = np.random.rand(dimension) < (
                    self.crossover_rate * np.sin(2 * np.pi * evaluations / self.budget)
                )
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimension)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    # Update memory by replacing the worst solution
                    mem_idx = np.argmax(memory_fitness)
                    if fitness[i] < memory_fitness[mem_idx]:
                        memory[mem_idx] = population[i]
                        memory_fitness[mem_idx] = fitness[i]

                    population[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
