import numpy as np


class ADAEDA:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 50
        self.initial_cr = 0.9
        self.initial_f = 0.8
        self.initial_temp = 1.0
        self.final_temp = 0.01
        self.alpha = 0.95  # Cooling rate

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutate(self, population, best_idx, cr, f):
        new_population = np.empty_like(population)
        for i in range(len(population)):
            idxs = np.random.choice(self.population_size, 3, replace=False)
            x1, x2, x3 = population[idxs[0]], population[idxs[1]], population[idxs[2]]
            mutant = population[best_idx] + f * (x1 - x2 + x3 - population[best_idx])
            new_population[i] = np.clip(mutant, self.bounds[0], self.bounds[1])
        return new_population

    def crossover(self, target, mutant, cr):
        crossover_mask = np.random.rand(self.dimension) < cr
        return np.where(crossover_mask, mutant, target)

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(population, func)
        evaluations = self.population_size
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_solution = population[best_idx]
        temperature = self.initial_temp
        cr = self.initial_cr
        f = self.initial_f

        while evaluations < self.budget:
            mutated_population = self.mutate(population, best_idx, cr, f)
            offspring_population = np.array(
                [
                    self.crossover(population[i], mutated_population[i], cr)
                    for i in range(self.population_size)
                ]
            )
            offspring_fitness = self.evaluate(offspring_population, func)
            evaluations += self.population_size

            for i in range(self.population_size):
                if offspring_fitness[i] < fitness[i]:
                    population[i], fitness[i] = offspring_population[i], offspring_fitness[i]
                    if fitness[i] < best_fitness:
                        best_fitness, best_solution, best_idx = fitness[i], population[i], i

            # Adaptive selective pressure based on temperature
            if temperature > self.final_temp:
                temperature *= self.alpha

        return best_fitness, best_solution
