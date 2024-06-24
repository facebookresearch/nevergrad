import numpy as np


class EADSEA:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 100
        self.elite_size = 10  # Number of elites

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutate(self, population, best_index, progress):
        mutation_factor = 0.5 + (1 - progress) * 0.5  # Adaptive mutation factor
        new_population = np.empty_like(population)
        for i in range(self.population_size):
            idxs = np.random.choice(np.delete(np.arange(self.population_size), best_index), 3, replace=False)
            a, b, c = population[idxs]
            mutant_vector = a + mutation_factor * (b - c)
            new_population[i] = np.clip(mutant_vector, self.bounds[0], self.bounds[1])
        return new_population

    def crossover(self, target, mutant, progress):
        crossover_rate = 0.7 + 0.2 * progress  # Adaptive crossover rate
        mask = np.random.rand(self.dimension) < crossover_rate
        return np.where(mask, mutant, target)

    def local_search(self, individual, func, iterations=10):
        step_size = 0.1
        for _ in range(iterations):
            candidate = individual + np.random.uniform(-step_size, step_size, self.dimension)
            candidate = np.clip(candidate, self.bounds[0], self.bounds[1])
            if func(candidate) < func(individual):
                individual = candidate
        return individual

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(population, func)
        evaluations = self.population_size
        best_index = np.argmin(fitness)

        while evaluations < self.budget:
            progress = evaluations / self.budget
            mutants = self.mutate(population, best_index, progress)
            trials = np.array(
                [self.crossover(population[i], mutants[i], progress) for i in range(self.population_size)]
            )
            fitness_trials = self.evaluate(trials, func)
            evaluations += len(trials)

            for i in range(self.population_size):
                if fitness_trials[i] < fitness[i]:
                    population[i] = trials[i]
                    fitness[i] = fitness_trials[i]

            new_best_index = np.argmin(fitness)
            if fitness[new_best_index] < fitness[best_index]:
                best_index = new_best_index

            # Elitism: preserve top performers
            elites_indices = np.argsort(fitness)[: self.elite_size]
            for i in elites_indices:
                population[i] = self.local_search(population[i], func)

        best_index = np.argmin(fitness)
        return fitness[best_index], population[best_index]
