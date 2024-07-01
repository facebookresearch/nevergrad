import numpy as np


class RDACE:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.initial_population_size = 100
        self.mutation_factor = 0.5
        self.crossover_rate = 0.7  # Adjusted for better exploration

    def initialize_population(self):
        return np.random.uniform(
            self.bounds[0], self.bounds[1], (self.initial_population_size, self.dimension)
        )

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutate(self, population, best_idx):
        mutants = np.empty_like(population)
        for i in range(len(population)):
            idxs = [idx for idx in range(len(population)) if idx != i]
            a, b, c = np.random.choice(idxs, 3, replace=False)
            mutant = population[a] + self.mutation_factor * (population[b] - population[c])
            mutants[i] = np.clip(mutant, self.bounds[0], self.bounds[1])
        return mutants

    def crossover(self, target, mutant):
        mask = np.random.rand(self.dimension) < self.crossover_rate
        return np.where(mask, mutant, target)

    def select(self, population, fitness, mutants, func):
        new_population = np.empty_like(population)
        new_fitness = np.empty_like(fitness)
        for i in range(len(population)):
            trial = self.crossover(population[i], mutants[i])
            trial_fitness = func(trial)
            if trial_fitness < fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness
            else:
                new_population[i] = population[i]
                new_fitness[i] = fitness[i]
        return new_population, new_fitness

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(population, func)
        evaluations = len(population)
        best_idx = np.argmin(fitness)

        while evaluations < self.budget:
            if evaluations + len(population) > self.budget:
                # Reduce population size to fit within budget
                excess = evaluations + len(population) - self.budget
                population = population[:-excess]
                fitness = fitness[:-excess]

            mutants = self.mutate(population, best_idx)
            population, fitness = self.select(population, fitness, mutants, func)
            evaluations += len(population)
            best_idx = np.argmin(fitness)

        best_individual = population[best_idx]
        best_fitness = func(best_individual)
        return best_fitness, best_individual
