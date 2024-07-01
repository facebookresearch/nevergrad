import numpy as np


class HEAS:
    def __init__(self, budget):
        self.budget = budget
        self.population_size = 20
        self.dimension = 5
        self.low = -5.0
        self.high = 5.0
        self.archive = []
        self.archive_max_size = 50  # reduced size to keep only relevant solutions

    def initialize(self):
        population = np.random.uniform(self.low, self.high, (self.population_size, self.dimension))
        F = np.random.normal(0.5, 0.1, self.population_size)
        CR = np.random.normal(0.9, 0.05, self.population_size)
        return population, F, CR

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutation(self, population, F):
        mutant = np.zeros_like(population)
        for i in range(self.population_size):
            indices = np.random.choice(self.population_size, 3, replace=False)
            x1, x2, x3 = population[indices]
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

        # Adapt F and CR with history-based adaptation
        F[improved] = np.clip(F[improved] * 1.1, 0.1, 1.0)
        CR[improved] = np.clip(CR[improved] * 0.95, 0.1, 1.0)
        F[~improved] = np.clip(F[~improved] * 0.9, 0.1, 1.0)
        CR[~improved] = np.clip(CR[~improved] * 1.05, 0.1, 1.0)

        return population, fitness, F, CR

    def local_search(self, individual, func):
        T = 1.0
        decay = 0.99
        for _ in range(10):
            neighbor = individual + np.random.normal(0, T, self.dimension)
            neighbor = np.clip(neighbor, self.low, self.high)
            if func(neighbor) < func(individual):
                individual = neighbor
            T *= decay
        return individual

    def __call__(self, func):
        population, F, CR = self.initialize()
        fitness = self.evaluate(population, func)
        iterations = self.budget // (self.population_size + 10)  # account for local searches

        for _ in range(iterations):
            mutant = self.mutation(population, F)
            trial_population = self.crossover(population, mutant, CR)
            trial_fitness = self.evaluate(trial_population, func)
            population, fitness, F, CR = self.select(
                population, fitness, trial_population, trial_fitness, F, CR
            )

            # Local search phase
            selected_indices = np.random.choice(self.population_size, size=5, replace=False)
            for idx in selected_indices:
                population[idx] = self.local_search(population[idx], func)
                fitness[idx] = func(population[idx])
                self.archive.append(population[idx].copy())  # update archive with locally searched solutions

            # Maintain a relevant archive
            if len(self.archive) > self.archive_max_size:
                self.archive = self.archive[-self.archive_max_size :]

        best_index = np.argmin(fitness)
        best_individual = population[best_index]
        best_fitness = fitness[best_index]
        return best_fitness, best_individual
