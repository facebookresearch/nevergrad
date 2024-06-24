import numpy as np


class AMDE:
    def __init__(
        self,
        budget,
        population_size=50,
        crossover_rate=0.8,
        F_base=0.5,
        F_amp=0.5,
        memory_size=20,
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

        # Initialize population uniformly within the bounds
        population = lb + (ub - lb) * np.random.rand(self.population_size, dimension)
        fitness = np.array([func(individual) for individual in population])

        # Memory for good solutions
        memory = population[: self.memory_size].copy()
        memory_fitness = fitness[: self.memory_size].copy()

        # Elite solutions tracking
        elite = population[: self.elite_size].copy()
        elite_fitness = fitness[: self.elite_size].copy()

        # Best solution tracking
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Adaptive mutation factor with a decaying amplitude
                F = self.F_base + self.F_amp * np.cos(2 * np.pi * evaluations / self.budget)

                # Mutation using memory, elite, or random selection
                if np.random.rand() < 0.2:  # Mutation from memory
                    m_idx = np.random.randint(self.memory_size)
                    a, b, c = memory[m_idx], population[np.random.randint(self.population_size)], elite[0]
                else:
                    idxs = np.random.choice(self.population_size, 3, replace=False)
                    a, b, c = population[idxs]

                mutant = np.clip(a + F * (b - c), lb, ub)

                # Crossover: Exponential
                start = np.random.randint(dimension)
                length = np.random.randint(1, dimension)
                cross_points = [(start + j) % dimension for j in range(length)]
                trial = population[i].copy()
                trial[cross_points] = mutant[cross_points]

                # Fitness evaluation and selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update memory if better
                    worst_memory_idx = np.argmax(memory_fitness)
                    if trial_fitness < memory_fitness[worst_memory_idx]:
                        memory[worst_memory_idx] = trial
                        memory_fitness[worst_memory_idx] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
