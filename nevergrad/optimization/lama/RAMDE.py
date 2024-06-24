import numpy as np


class RAMDE:
    def __init__(
        self,
        budget,
        population_size=50,
        crossover_rate=0.9,
        F_base=0.6,
        F_amp=0.4,
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

        # Initialize population uniformly within the bounds
        population = lb + (ub - lb) * np.random.rand(self.population_size, dimension)
        fitness = np.array([func(individual) for individual in population])

        # Memory initialization
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
            # Adjust mutation factor with decaying amplitude and sine function
            F = self.F_base + self.F_amp * np.sin(2 * np.pi * evaluations / self.budget)

            for i in range(self.population_size):
                # Select individuals for mutation
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                # Mutation: Incorporate both best and random elite solutions
                best_or_elite = (
                    best_solution if np.random.rand() < 0.75 else elite[np.random.randint(self.elite_size)]
                )
                mutant = np.clip(a + F * (best_or_elite - b + c - a), lb, ub)

                # Crossover: Binomial
                cross_points = np.random.rand(dimension) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(dimension)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Fitness evaluation and selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    # Update memory if better
                    worst_memory_idx = np.argmax(memory_fitness)
                    if trial_fitness < memory_fitness[worst_memory_idx]:
                        memory[worst_memory_idx] = trial
                        memory_fitness[worst_memory_idx] = trial_fitness

                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update elite if better
                    worst_elite_idx = np.argmax(elite_fitness)
                    if trial_fitness < elite_fitness[worst_elite_idx]:
                        elite[worst_elite_idx] = trial
                        elite_fitness[worst_elite_idx] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
