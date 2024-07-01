import numpy as np


class EADES:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 50
        self.cr = 0.9  # Crossover probability
        self.f = 0.8  # Differential weight

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutate(self, population):
        new_population = np.empty_like(population)
        for i in range(len(population)):
            idxs = np.random.choice(self.population_size, 3, replace=False)
            x1, x2, x3 = population[idxs]
            mutant = np.clip(x1 + self.f * (x2 - x3), self.bounds[0], self.bounds[1])
            new_population[i] = mutant
        return new_population

    def crossover(self, target, mutant):
        crossover_mask = np.random.rand(self.dimension) < self.cr
        return np.where(crossover_mask, mutant, target)

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(population, func)
        evaluations = self.population_size
        best_fitness = np.min(fitness)
        best_solution = population[np.argmin(fitness)]

        while evaluations < self.budget:
            mutated_population = self.mutate(population)
            offspring_population = np.array(
                [self.crossover(population[i], mutated_population[i]) for i in range(self.population_size)]
            )
            offspring_fitness = self.evaluate(offspring_population, func)
            evaluations += self.population_size

            for i in range(self.population_size):
                if offspring_fitness[i] < fitness[i]:
                    population[i], fitness[i] = offspring_population[i], offspring_fitness[i]
                    if fitness[i] < best_fitness:
                        best_fitness, best_solution = fitness[i], population[i]

            # Adaptive differential weight adjustment
            self.f *= 0.995  # Gradual decrease to focus more on exploration initially and exploitation later
            if evaluations % (self.budget // 10) == 0:
                self.f = max(self.f, 0.5)  # Prevent it from becoming too small

        return best_fitness, best_solution
