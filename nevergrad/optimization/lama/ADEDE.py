import numpy as np


class ADEDE:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 100
        self.mutation_factor = 0.8
        self.crossover_probability = 0.9
        self.adaptation_rate = 0.05

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutate(self, population, best_idx, diversity):
        mutants = np.empty_like(population)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = np.random.choice(idxs, 3, replace=False)
            mutant = population[a] + self.mutation_factor * (population[b] - population[c])

            # Leverages diversity to adjust mutation
            mutant = mutant + diversity * (population[best_idx] - mutant)
            mutant = np.clip(mutant, self.bounds[0], self.bounds[1])
            mutants[i] = mutant
        return mutants

    def crossover(self, target, mutant):
        mask = np.random.rand(self.dimension) < self.crossover_probability
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
        best_idx = np.argmin(fitness)

        while evaluations < self.budget:
            diversity = np.std(population, axis=0).mean() / (self.bounds[1] - self.bounds[0])
            mutants = self.mutate(population, best_idx, diversity)
            population, fitness = self.select(population, fitness, mutants, func)
            evaluations += self.population_size
            best_idx = np.argmin(fitness)

            # Update mutation factor based on diversity
            self.mutation_factor = np.clip(self.mutation_factor - self.adaptation_rate + diversity, 0.1, 1.0)

        best_index = np.argmin(fitness)
        return fitness[best_index], population[best_index]
