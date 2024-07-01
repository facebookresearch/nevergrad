import numpy as np


class DAESF:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 100
        self.initial_population_size = 100
        self.min_population_size = 50
        self.max_population_size = 200
        self.decrease_factor = 0.9
        self.increase_factor = 1.1

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutate(self, population, best, F):
        new_population = np.empty_like(population)
        for i in range(len(population)):
            idxs = np.random.choice(np.arange(len(population)), 3, replace=False)
            a, b, c = population[idxs]
            mutant_vector = np.clip(a + F * (b - c), self.bounds[0], self.bounds[1])
            new_population[i] = mutant_vector
        return new_population

    def crossover(self, target, mutant, CR):
        mask = np.random.rand(self.dimension) < CR
        return np.where(mask, mutant, target)

    def adjust_population_size(self, improvement):
        if improvement < 0.01:
            self.population_size = min(
                self.max_population_size, int(self.population_size * self.increase_factor)
            )
        else:
            self.population_size = max(
                self.min_population_size, int(self.population_size * self.decrease_factor)
            )

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(population, func)
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        best_fitness = fitness[best_index]

        while evaluations < self.budget:
            F = np.random.normal(0.5, 0.1)
            CR = 0.1 + 0.4 * np.random.rand()
            mutants = self.mutate(population, population[best_index], F)
            trials = np.array(
                [self.crossover(population[i], mutants[i], CR) for i in range(self.population_size)]
            )
            fitness_trials = self.evaluate(trials, func)
            evaluations += self.population_size

            improvement = False
            for i in range(self.population_size):
                if fitness_trials[i] < fitness[i]:
                    population[i] = trials[i]
                    fitness[i] = fitness_trials[i]
                    if fitness[i] < best_fitness:
                        best_fitness = fitness[i]
                        best_index = i
                        improvement = True

            if not improvement:
                population = np.vstack((population, self.initialize_population()))
                fitness = self.evaluate(population, func)
                best_index = np.argmin(fitness)
                best_fitness = fitness[best_index]
                evaluations += self.population_size

            self.adjust_population_size(np.abs(fitness[best_index] - best_fitness))

        return best_fitness, population[best_index]
