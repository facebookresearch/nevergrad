import numpy as np


class ADCE:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 100
        self.mutation_base = 0.5
        self.crossover_base = 0.7

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutate(self, population, best_idx):
        mutants = np.empty_like(population)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i and idx != best_idx]
            a, b, c = np.random.choice(idxs, 3, replace=False)
            mutation_factor = self.mutation_base + np.random.rand() * (1.0 - self.mutation_base)
            mutant = population[a] + mutation_factor * (population[b] - population[c])
            mutants[i] = np.clip(mutant, self.bounds[0], self.bounds[1])
        return mutants

    def crossover(self, target, mutant):
        crossover_prob = self.crossover_base + np.random.rand() * (1.0 - self.crossover_base)
        mask = np.random.rand(self.dimension) < crossover_prob
        return np.where(mask, mutant, target)

    def select(self, population, fitness, mutants, func):
        new_population = np.empty_like(population)
        new_fitness = np.empty_like(fitness)
        for i in range(self.population_size):
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
        evaluations = self.population_size

        while evaluations < self.budget:
            best_idx = np.argmin(fitness)
            mutants = self.mutate(population, best_idx)
            population, fitness = self.select(population, fitness, mutants, func)
            evaluations += self.population_size

        best_index = np.argmin(fitness)
        return fitness[best_index], population[best_index]
