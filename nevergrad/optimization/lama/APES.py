import numpy as np


class APES:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 50
        self.mutation_factor = 0.5
        self.crossover_prob = 0.7

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutate(self, population, best_idx, scaling_factor):
        new_population = np.empty_like(population)
        for i in range(len(population)):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + scaling_factor * (b - c), self.bounds[0], self.bounds[1])
            new_population[i] = mutant
        return new_population

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dimension) < self.crossover_prob
        trial = np.where(cross_points, mutant, target)
        return trial

    def select(self, population, fitness, new_population, new_fitness):
        for i in range(self.population_size):
            if new_fitness[i] < fitness[i]:
                population[i], fitness[i] = new_population[i], new_fitness[i]

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(population, func)
        evaluations = self.population_size
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_solution = population[best_idx]

        while evaluations < self.budget:
            scaling_factor = self.mutation_factor / (1 + np.exp(-0.1 * (evaluations / self.budget - 5)))
            mutated_population = self.mutate(population, best_idx, scaling_factor)
            trial_population = np.array(
                [self.crossover(population[i], mutated_population[i]) for i in range(self.population_size)]
            )
            trial_fitness = self.evaluate(trial_population, func)
            evaluations += self.population_size

            self.select(population, fitness, trial_population, trial_fitness)
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_fitness, best_solution = fitness[best_idx], population[best_idx]

        return best_fitness, best_solution
