import numpy as np


class DNAS:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 100
        self.learning_rate = 0.1
        self.mutation_scale = 0.8

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutate(self, population, best, worst, F):
        new_population = np.empty_like(population)
        for i in range(len(population)):
            idxs = np.random.choice(np.arange(len(population)), 2, replace=False)
            x1, x2 = population[idxs]
            mutant_vector = np.clip(
                x1 + F * (x2 - population[i]) + self.learning_rate * (best - worst),
                self.bounds[0],
                self.bounds[1],
            )
            new_population[i] = mutant_vector
        return new_population

    def crossover(self, target, mutant, CR):
        mask = np.random.rand(self.dimension) < CR
        return np.where(mask, mutant, target)

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(population, func)
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        best_fitness = fitness[best_index]
        previous_best = best_fitness

        while evaluations < self.budget:
            F = np.random.normal(self.mutation_scale, 0.1) * (1 + 0.1 * np.random.rand())
            CR = 0.1 + 0.5 * np.random.rand()
            worst_index = np.argmax(fitness)
            mutants = self.mutate(population, population[best_index], population[worst_index], F)
            trials = np.array(
                [self.crossover(population[i], mutants[i], CR) for i in range(self.population_size)]
            )
            fitness_trials = self.evaluate(trials, func)
            evaluations += self.population_size

            for i in range(self.population_size):
                if fitness_trials[i] < fitness[i]:
                    population[i] = trials[i]
                    fitness[i] = fitness_trials[i]
                    if fitness[i] < best_fitness:
                        best_fitness = fitness[i]
                        best_index = i

            if best_fitness < previous_best:
                self.learning_rate *= 1.1
                previous_best = best_fitness
            else:
                self.learning_rate *= 0.9
                if np.random.rand() < 0.1:
                    population[
                        np.random.choice(len(population), size=int(self.population_size * 0.1), replace=False)
                    ] = np.random.uniform(
                        self.bounds[0], self.bounds[1], (int(self.population_size * 0.1), self.dimension)
                    )

        return best_fitness, population[best_index]
