import numpy as np


class DAES:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 50
        self.cr = 0.9  # Initial crossover probability
        self.f = 0.8  # Initial differential weight

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutate(self, population, fitness):
        new_population = np.empty_like(population)
        best_idx = np.argmin(fitness)
        for i in range(len(population)):
            idxs = np.random.choice(self.population_size, 4, replace=False)
            x1, x2, x3, x4 = (
                population[idxs[0]],
                population[idxs[1]],
                population[idxs[2]],
                population[idxs[3]],
            )
            if np.random.rand() < 0.5:  # Randomly choose mutation strategy
                mutant = x1 + self.f * (x2 - x3 + x4 - population[i])  # DE/rand/2
            else:
                mutant = population[i] + self.f * (x1 - x2 + x3 - x4)  # DE/current-to-rand/1
            new_population[i] = np.clip(mutant, self.bounds[0], self.bounds[1])
        return new_population

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dimension) < self.cr
        return np.where(crossover_mask, mutant, target)

    def adapt_parameters(self, improvements):
        if improvements > 0:
            self.cr = max(0.1, self.cr * 0.98)
            self.f = max(0.5, self.f * 0.99)
        else:
            self.cr = min(1.0, self.cr / 0.95)
            self.f = min(1.2, self.f / 0.95)

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(population, func)
        evaluations = self.population_size
        best_fitness = np.min(fitness)
        best_solution = population[np.argmin(fitness)]
        last_best = best_fitness

        while evaluations < self.budget:
            mutated_population = self.mutate(population, fitness)
            offspring_population = np.array(
                [self.crossover(population[i], mutated_population[i]) for i in range(self.population_size)]
            )
            offspring_fitness = self.evaluate(offspring_population, func)
            evaluations += self.population_size

            improvements = 0
            for i in range(self.population_size):
                if offspring_fitness[i] < fitness[i]:
                    population[i], fitness[i] = offspring_population[i], offspring_fitness[i]
                    improvements += 1
                    if fitness[i] < best_fitness:
                        best_fitness, best_solution = fitness[i], population[i]

            self.adapt_parameters(improvements)
            if best_fitness == last_best:
                improvements = 0
            else:
                last_best = best_fitness

        return best_fitness, best_solution
