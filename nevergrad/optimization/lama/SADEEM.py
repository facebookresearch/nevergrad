import numpy as np


class SADEEM:
    def __init__(self, budget, population_size=30, F=0.8, CR=0.9):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.F = F
        self.CR = CR

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dimension))

    def evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutation(self, population, best_idx):
        new_population = np.zeros_like(population)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = np.random.choice(idxs, 3, replace=False)
            mutant = population[a] + self.F * (population[b] - population[c])
            # Simple OBL
            if np.random.rand() < 0.1:
                mutant = self.lower_bound + self.upper_bound - mutant
            new_population[i] = np.clip(mutant, self.lower_bound, self.upper_bound)
        return new_population

    def crossover(self, population, mutant_population):
        crossover_population = np.array(
            [
                np.where(np.random.rand(self.dimension) < self.CR, mutant_population[i], population[i])
                for i in range(self.population_size)
            ]
        )
        return crossover_population

    def select(self, population, fitness, trial_population, func):
        trial_fitness = self.evaluate_population(trial_population, func)
        for i in range(self.population_size):
            if trial_fitness[i] < fitness[i]:
                fitness[i] = trial_fitness[i]
                population[i] = trial_population[i]
        return population, fitness

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate_population(population, func)
        best_idx = np.argmin(fitness)

        evaluations = self.population_size
        while evaluations < self.budget:
            mutant_population = self.mutation(population, best_idx)
            crossover_population = self.crossover(population, mutant_population)
            population, fitness = self.select(population, fitness, crossover_population, func)
            evaluations += self.population_size
            best_idx = np.argmin(fitness)

        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        return best_fitness, best_solution
