import numpy as np


class EIADEA:
    def __init__(
        self, budget, population_size=40, crossover_rate=0.85, F_min=0.5, F_max=0.8, archive_size=40
    ):
        self.budget = budget
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.F_min = F_min
        self.F_max = F_max
        self.archive_size = archive_size

    def __call__(self, func):
        # Define the bounds and the dimensionality of the problem
        lb = -5.0
        ub = 5.0
        dimension = 5

        # Initialize population with random values within the bounds
        population = lb + (ub - lb) * np.random.rand(self.population_size, dimension)
        fitness = np.array([func(individual) for individual in population])

        # Initialize an archive to store potential solutions
        archive = np.empty((0, dimension))

        # Track the best solution found so far
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                # Adaptive mutation factor decreases as iterations progress
                F = np.random.uniform(self.F_min, self.F_max) * (1 - evaluations / self.budget)

                # Selection of mutation vectors: Ensure diversity by picking distinct indices
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                # Mutation using DE/rand/1/bin strategy with archive incorporation for diversity
                if archive.size > 0 and np.random.rand() < 0.15:
                    arch_idx = np.random.randint(0, archive.shape[0])
                    a = archive[arch_idx]
                mutant = np.clip(a + F * (b - c), lb, ub)

                # Crossover: Combine mutant with target vector
                cross_points = np.random.rand(dimension) < self.crossover_rate
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, dimension)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Evaluate the new candidate solution
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < fitness[i]:
                    # Update archive with the replaced solution if the archive is not full
                    if archive.shape[0] < self.archive_size:
                        archive = np.vstack([archive, population[i]])
                    else:
                        # Replace a randomly selected entry in the archive
                        archive[np.random.randint(0, self.archive_size)] = population[i]

                    # Update the population with the new better solution
                    population[i] = trial
                    fitness[i] = trial_fitness

                    # Update the best solution if the new solution is better
                    if trial_fitness < best_fitness:
                        best_solution = trial
                        best_fitness = trial_fitness

                # Exit if the budget is exhausted
                if evaluations >= self.budget:
                    break

        return best_fitness, best_solution
