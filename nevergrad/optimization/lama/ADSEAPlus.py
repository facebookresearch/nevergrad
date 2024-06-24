import numpy as np


class ADSEAPlus:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 100
        self.archive_size = 20
        self.initial_mutation_factor = 0.8
        self.initial_crossover_prob = 0.7
        self.archive = []

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def adapt_parameters(self, progress):
        # Adapt the mutation factor and crossover probability as the search progresses
        self.mutation_factor = self.initial_mutation_factor * (1 - progress)
        self.crossover_prob = self.initial_crossover_prob + progress * (0.9 - self.initial_crossover_prob)

    def mutate(self, population, best, func, progress):
        self.adapt_parameters(progress)
        new_population = np.empty_like(population)
        for i in range(self.population_size):
            idxs = np.random.choice(len(population), 4, replace=False)
            mutated = (
                best
                + self.mutation_factor * (population[idxs[0]] - population[idxs[1]])
                + self.mutation_factor * (population[idxs[2]] - population[idxs[3]])
            )
            new_population[i] = np.clip(mutated, self.bounds[0], self.bounds[1])
        return new_population

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dimension) < self.crossover_prob
        return np.where(cross_points, mutant, target)

    def select(self, population, trials, fitness_trials):
        for i in range(self.population_size):
            if fitness_trials[i] < self.fitness[i]:
                population[i] = trials[i]
                self.fitness[i] = fitness_trials[i]

    def update_archive(self, population, fitness):
        combined = list(zip(self.archive + list(population), list(self.fitness) + list(fitness)))
        combined.sort(key=lambda x: x[1])
        self.archive = [x[0] for x in combined[: self.archive_size]]

    def __call__(self, func):
        population = self.initialize_population()
        self.fitness = self.evaluate(population, func)
        evaluations = self.population_size

        while evaluations < self.budget:
            best_idx = np.argmin(self.fitness)
            best = population[best_idx]
            progress = evaluations / self.budget

            mutants = self.mutate(population, best, func, progress)
            trials = np.array(
                [self.crossover(population[i], mutants[i]) for i in range(self.population_size)]
            )
            fitness_trials = self.evaluate(trials, func)
            self.select(population, trials, fitness_trials)
            evaluations += len(trials)

            self.update_archive(population, self.fitness)

            if evaluations + self.population_size > self.budget:
                break

        best_idx = np.argmin(self.fitness)
        return self.fitness[best_idx], population[best_idx]
