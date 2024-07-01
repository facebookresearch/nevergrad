import numpy as np


class EORAMED:
    def __init__(
        self,
        budget,
        population_size=50,
        crossover_rate=0.9,
        F_min=0.5,
        F_max=0.9,
        memory_size=50,
        elite_size=5,
    ):
        self.budget = budget
        self.population_size = population_size
        self.crossover_rate = crossover_rate  # Lowered to encourage more diversity
        self.F_min = F_min
        self.F_max = F_max
        self.memory_size = memory_size
        self.elite_size = elite_size  # Reduced to concentrate on fewer, high-quality elites

    def __call__(self, func):
        lb, ub, dimension = -5.0, 5.0, 5

        # Initialize population randomly
        population = lb + (ub - lb) * np.random.rand(self.population_size, dimension)
        fitness = np.array([func(individual) for individual in population])

        # Memory for good solutions
        memory = np.empty((self.memory_size, dimension))
        memory_fitness = np.full(self.memory_size, np.inf)

        # Elite solutions
        elite = np.empty((self.elite_size, dimension))
        elite_fitness = np.full(self.elite_size, np.inf)

        # Track best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            # Update elite pool
            elite_indices = np.argsort(fitness)[: self.elite_size]
            elite = population[elite_indices].copy()
            elite_fitness = fitness[elite_indices].copy()

            for i in range(self.population_size):
                # Adaptive mutation factor with linear decay
                F = self.F_max - (evaluations / self.budget) * (self.F_max - self.F_min)

                # Mutation with higher emphasis on best and elite
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                best_or_elite = (
                    best_solution if np.random.rand() < 0.7 else elite[np.random.randint(0, self.elite_size)]
                )
                mutant = np.clip(a + F * (best_or_elite - a + b - c), lb, ub)

                # Crossover
                cross_points = np.random.rand(dimension) < self.crossover_rate
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate and possibly update population
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
