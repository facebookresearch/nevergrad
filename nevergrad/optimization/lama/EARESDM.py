import numpy as np


class EARESDM:
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

        # Initialize population and fitness
        population = lb + (ub - lb) * np.random.rand(self.population_size, dimension)
        fitness = np.array([func(individual) for individual in population])

        # Initialize memory for good solutions and their fitness
        memory = np.empty((self.memory_size, dimension))
        memory_fitness = np.full(self.memory_size, np.inf)

        # Initialize elite solutions and their fitness
        elite = np.empty((self.elite_size, dimension))
        elite_fitness = np.full(self.elite_size, np.inf)

        # Track the best solution and its fitness
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            # Periodic update of elite solutions
            if evaluations % (self.budget // 20) == 0:
                elite_idxs = np.argsort(fitness)[: self.elite_size]
                elite = population[elite_idxs].copy()
                elite_fitness = fitness[elite_idxs].copy()

            for i in range(self.population_size):
                # Adaptive mutation factor with directional influence
                F = self.F_base + self.F_amp * np.sin(2 * np.pi * evaluations / self.budget)

                # Mutation: differential mutation with directional influence from elite
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                random_elite = elite[np.random.randint(self.elite_size)]
                mutant = np.clip(a + F * (random_elite - a + b - c), lb, ub)

                # Adaptive crossover
                cross_points = np.random.rand(dimension) < self.crossover_rate + 0.05 * np.cos(
                    2 * np.pi * evaluations / self.budget
                )
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate and select
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    # Memory update: replace the worst entry if the current individual is better
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
