import numpy as np


class ASADEA:
    def __init__(self, budget):
        self.budget = budget
        self.population_size = 20
        self.dimension = 5
        self.low = -5.0
        self.high = 5.0
        self.archive = []
        self.archive_max_size = 100

    def initialize(self):
        population = np.random.uniform(self.low, self.high, (self.population_size, self.dimension))
        # Initialize F and CR for each individual
        F = np.random.normal(0.5, 0.1, self.population_size)
        CR = np.random.normal(0.9, 0.05, self.population_size)
        return population, F, CR

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutation(self, population, archive, F):
        mutant = np.zeros_like(population)
        combined = np.vstack([population] + [archive]) if archive else population
        num_candidates = len(combined)

        for i in range(self.population_size):
            indices = np.random.choice(num_candidates, 3, replace=False)
            x1, x2, x3 = combined[indices]
            mutant_vector = x1 + F[i] * (x2 - x3)
            mutant[i] = np.clip(mutant_vector, self.low, self.high)
        return mutant

    def crossover(self, population, mutant, CR):
        crossover = np.where(
            np.random.rand(self.population_size, self.dimension) < CR[:, None], mutant, population
        )
        return crossover

    def select(self, population, fitness, trial_population, trial_fitness, F, CR):
        improved = trial_fitness < fitness
        population[improved] = trial_population[improved]
        fitness[improved] = trial_fitness[improved]
        # Adapt F and CR
        F[improved] *= 1.1
        CR[improved] *= 0.95
        F[~improved] *= 0.9
        CR[~improved] *= 1.05
        F = np.clip(F, 0.1, 1.0)
        CR = np.clip(CR, 0.1, 1.0)
        return population, fitness, F, CR

    def __call__(self, func):
        population, F, CR = self.initialize()
        fitness = self.evaluate(population, func)
        iterations = self.budget // self.population_size

        for _ in range(iterations):
            if np.random.rand() < 0.1:  # Introduce perturbation with 10% chance
                random_archive_idx = np.random.choice(len(self.archive)) if self.archive else 0
                population += np.random.normal(0, 0.1) * self.archive[random_archive_idx]

            mutant = self.mutation(population, self.archive, F)
            trial_population = self.crossover(population, mutant, CR)
            trial_fitness = self.evaluate(trial_population, func)
            population, fitness, F, CR = self.select(
                population, fitness, trial_population, trial_fitness, F, CR
            )

            # Update the archive with new solutions
            for ind in population:
                self.archive.append(ind.copy())
                if len(self.archive) > self.archive_max_size:
                    self.archive.pop(0)

        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]
        return best_fitness, best_individual
