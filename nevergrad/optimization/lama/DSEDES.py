import numpy as np


class DSEDES:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 100
        self.elite_size = 10
        self.mutation_factor = 0.15
        self.crossover_prob = 0.7

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutate(self, population, best_index, current_index):
        indices = [i for i in range(self.population_size) if i != current_index]
        a, b, c = np.random.choice(indices, 3, replace=False)
        mutant = population[a] + self.mutation_factor * (population[b] - population[c])
        return np.clip(mutant, self.bounds[0], self.bounds[1])

    def crossover(self, target, mutant):
        mask = np.random.rand(self.dimension) < self.crossover_prob
        trial = np.where(mask, mutant, target)
        return trial

    def select(self, target, trial, target_fitness, trial_fitness):
        if trial_fitness < target_fitness:
            return trial, trial_fitness
        else:
            return target, target_fitness

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(population, func)
        best_index = np.argmin(fitness)
        best_individual = population[best_index].copy()
        best_fitness = fitness[best_index]

        evaluations = self.population_size
        while evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self.mutate(population, best_index, i)
                trial = self.crossover(population[i], mutant)
                trial_fitness = func(trial)
                evaluations += 1

                population[i], fitness[i] = self.select(population[i], trial, fitness[i], trial_fitness)

                if fitness[i] < best_fitness:
                    best_fitness = fitness[i]
                    best_individual = population[i].copy()

            if evaluations % 100 == 0:
                self.mutation_factor = np.clip(
                    self.mutation_factor * (0.95 if np.random.rand() < 0.5 else 1.05), 0.05, 1
                )
                self.crossover_prob = np.clip(
                    self.crossover_prob + (0.05 if np.random.rand() < 0.5 else -0.05), 0.1, 0.9
                )

        return best_fitness, best_individual
