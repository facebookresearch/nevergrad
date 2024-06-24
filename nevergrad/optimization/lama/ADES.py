import numpy as np


class ADES:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.initial_population_size = 100
        self.min_mutation_factor = 0.1
        self.max_mutation_factor = 0.9
        self.min_crossover_rate = 0.1
        self.max_crossover_rate = 0.9

    def initialize_population(self):
        return np.random.uniform(
            self.bounds[0], self.bounds[1], (self.initial_population_size, self.dimension)
        )

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutate(self, population, best_idx, success_rate):
        mutation_factor = (
            self.min_mutation_factor + (self.max_mutation_factor - self.min_mutation_factor) * success_rate
        )
        mutants = np.empty_like(population)
        for i in range(len(population)):
            idxs = [idx for idx in range(len(population)) if idx != i]
            a, b, c = np.random.choice(idxs, 3, replace=False)
            mutant = population[a] + mutation_factor * (population[b] - population[c])
            mutants[i] = np.clip(mutant, self.bounds[0], self.bounds[1])
        return mutants

    def crossover(self, target, mutant, success_rate):
        crossover_rate = (
            self.min_crossover_rate + (self.max_crossover_rate - self.min_crossover_rate) * success_rate
        )
        mask = np.random.rand(self.dimension) < crossover_rate
        return np.where(mask, mutant, target)

    def select(self, population, fitness, mutants, func):
        new_population = np.empty_like(population)
        new_fitness = np.empty_like(fitness)
        successful_trials = 0
        for i in range(len(population)):
            trial = self.crossover(population[i], mutants[i], successful_trials / max(1, i))
            trial_fitness = func(trial)
            if trial_fitness < fitness[i]:
                new_population[i] = trial
                new_fitness[i] = trial_fitness
                successful_trials += 1
            else:
                new_population[i] = population[i]
                new_fitness[i] = fitness[i]
        success_rate = successful_trials / len(population)
        return new_population, new_fitness, success_rate

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(population, func)
        evaluations = len(population)
        best_idx = np.argmin(fitness)
        success_rate = 0.5  # Start with a neutral success rate

        while evaluations < self.budget:
            if evaluations + len(population) > self.budget:
                # Reduce population size to fit within budget
                excess = evaluations + len(population) - self.budget
                population = population[:-excess]
                fitness = fitness[:-excess]
            mutants = self.mutate(population, best_idx, success_rate)
            population, fitness, success_rate = self.select(population, fitness, mutants, func)
            evaluations += len(population)
            best_idx = np.argmin(fitness)

        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]
        return best_fitness, best_individual
