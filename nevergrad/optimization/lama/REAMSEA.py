import numpy as np


class REAMSEA:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 100
        self.archive = []
        self.archive_size = 50

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutate(self, population, archive, best_index):
        F = np.random.normal(0.5, 0.1)
        new_population = np.empty_like(population)
        combined = np.vstack((population, archive))
        for i in range(self.population_size):
            idxs = np.random.choice(np.arange(len(combined)), 3, replace=False)
            a, b, c = combined[idxs]
            mutant_vector = a + F * (b - c)
            new_population[i] = np.clip(mutant_vector, self.bounds[0], self.bounds[1])
        return new_population

    def crossover(self, target, mutant):
        CR = 0.1 + 0.4 * np.random.rand()
        mask = np.random.rand(self.dimension) < CR
        return np.where(mask, mutant, target)

    def local_search(self, best_candidate, func):
        step_size = 0.1
        local_best = best_candidate
        local_best_fitness = func(best_candidate)
        for _ in range(10):
            candidate = local_best + np.random.uniform(-step_size, step_size, self.dimension)
            candidate = np.clip(candidate, self.bounds[0], self.bounds[1])
            candidate_fitness = func(candidate)
            if candidate_fitness < local_best_fitness:
                local_best = candidate
                local_best_fitness = candidate_fitness
        return local_best

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(population, func)
        evaluations = self.population_size
        best_index = np.argmin(fitness)

        while evaluations < self.budget:
            if len(self.archive) > self.archive_size:
                self.archive.pop(0)
            self.archive.append(population[best_index])

            mutants = self.mutate(population, self.archive, best_index)
            trials = np.array(
                [self.crossover(population[i], mutants[i]) for i in range(self.population_size)]
            )
            fitness_trials = self.evaluate(trials, func)
            evaluations += len(trials)

            for i in range(self.population_size):
                if fitness_trials[i] < fitness[i]:
                    population[i] = trials[i]
                    fitness[i] = fitness_trials[i]

            best_index = np.argmin(fitness)

            if evaluations % 100 == 0:
                population[best_index] = self.local_search(population[best_index], func)

        return fitness[best_index], population[best_index]
