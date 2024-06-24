import numpy as np


class MADE:
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

    def select_mutation_strategy(self):
        strategies = ["rand", "best", "current_to_best"]
        return np.random.choice(strategies)

    def mutate(self, population, fitness):
        strategy = self.select_mutation_strategy()
        new_population = np.empty_like(population)
        best_idx = np.argmin(fitness)
        for i in range(len(population)):
            idxs = np.random.choice(self.population_size, 3, replace=False)
            x1, x2, x3, x_best = (
                population[idxs[0]],
                population[idxs[1]],
                population[idxs[2]],
                population[best_idx],
            )
            if strategy == "rand":
                mutant = x1 + self.f * (x2 - x3)
            elif strategy == "best":
                mutant = x_best + self.f * (x1 - x2)
            elif strategy == "current_to_best":
                mutant = population[i] + self.f * (x_best - population[i] + x1 - x2)
            new_population[i] = np.clip(mutant, self.bounds[0], self.bounds[1])
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
            mutated_population = self.mutate(population, fitness)
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

            self.adapt_parameters()

        return best_fitness, best_solution

    def adapt_parameters(self):
        self.cr = max(0.5, self.cr * 0.98)  # Adaptively decrease CR
        self.f = max(0.5, self.f * 0.99)  # Adaptively decrease F if no improvement
