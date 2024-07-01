import numpy as np


class ADEA:
    def __init__(self, budget, population_size=30, crossover_rate=0.8, F=0.5, archive_size=50):
        self.budget = budget
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.F = F
        self.archive_size = archive_size

    def __call__(self, func):
        # Bounds and dimensionality
        lb = -5.0
        ub = 5.0
        dimension = 5

        # Initialize population
        population = lb + (ub - lb) * np.random.rand(self.population_size, dimension)
        fitness = np.array([func(individual) for individual in population])

        # Initialize archive
        archive = np.empty((0, dimension))

        # Best solution found
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx, :]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size

        # Main loop
        while evaluations < self.budget:
            for i in range(self.population_size):
                # Mutation (DE/rand/1/bin)
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)

                # Crossover
                cross_points = np.random.rand(dimension) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimension)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    # Update archive
                    if archive.shape[0] < self.archive_size:
                        archive = np.vstack([archive, population[i]])
                    else:
                        archive[np.random.randint(self.archive_size)] = population[i]

                    # Update population
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update best solution
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
