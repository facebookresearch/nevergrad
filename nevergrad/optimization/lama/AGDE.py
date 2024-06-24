import numpy as np


class AGDE:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 100
        self.elite_size = 5
        self.mutation_factor = 0.5
        self.crossover_prob = 0.7

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutate(self, population, best_index):
        new_population = np.empty_like(population)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = np.random.choice(idxs, 3, replace=False)
            mutant_vector = population[a] + self.mutation_factor * (population[b] - population[c])
            new_population[i] = np.clip(mutant_vector, self.bounds[0], self.bounds[1])
        return new_population

    def crossover(self, target, mutant):
        cross_points = np.random.rand(self.dimension) < self.crossover_prob
        trial = np.where(cross_points, mutant, target)
        return trial

    def select(self, population, trials, fitness_trials):
        for i in range(self.population_size):
            if fitness_trials[i] < self.fitness[i]:
                population[i] = trials[i]
                self.fitness[i] = fitness_trials[i]

    def enhance_elites(self, population, func):
        elites_indices = np.argsort(self.fitness)[: self.elite_size]
        for idx in elites_indices:
            local_search_vector = population[idx] + np.random.normal(0, 0.1, self.dimension)
            local_search_vector = np.clip(local_search_vector, self.bounds[0], self.bounds[1])
            f_local = func(local_search_vector)
            if f_local < self.fitness[idx]:
                population[idx] = local_search_vector
                self.fitness[idx] = f_local

    def __call__(self, func):
        population = self.initialize_population()
        self.fitness = self.evaluate(population, func)
        evaluations = self.population_size

        while evaluations < self.budget:
            mutants = self.mutate(population, np.argmin(self.fitness))
            trials = np.array(
                [self.crossover(population[i], mutants[i]) for i in range(self.population_size)]
            )
            fitness_trials = self.evaluate(trials, func)
            self.select(population, trials, fitness_trials)
            evaluations += self.population_size

            if evaluations + self.population_size > self.budget:
                break

            if evaluations % 100 == 0:
                self.enhance_elites(population, func)
                self.mutation_factor *= 0.98
                self.crossover_prob = np.clip(
                    self.crossover_prob * (0.99 if np.random.rand() < 0.5 else 1.01), 0.1, 0.9
                )

        best_idx = np.argmin(self.fitness)
        return self.fitness[best_idx], population[best_idx]
