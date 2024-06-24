import numpy as np


class DDCEA:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 100
        self.elite_size = 5  # Reduced number of elites for more diversity

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutate(self, population, best_index, diversity_measure):
        F = 0.5 * (2 - diversity_measure)  # Adaptive mutation factor based on diversity
        new_population = np.empty_like(population)
        for i in range(self.population_size):
            idxs = np.random.choice(np.delete(np.arange(self.population_size), best_index), 3, replace=False)
            a, b, c = population[idxs]
            mutant_vector = a + F * (b - c)
            new_population[i] = np.clip(mutant_vector, self.bounds[0], self.bounds[1])
        return new_population

    def crossover(self, target, mutant, diversity_measure):
        CR = 0.5 + 0.5 * diversity_measure  # Higher crossover when diversity is high
        mask = np.random.rand(self.dimension) < CR
        return np.where(mask, mutant, target)

    def calculate_diversity(self, population):
        mean_population = np.mean(population, axis=0)
        diversity = np.mean(np.sqrt(np.sum((population - mean_population) ** 2, axis=1)))
        normalized_diversity = diversity / np.sqrt(self.dimension * (self.bounds[1] - self.bounds[0]) ** 2)
        return normalized_diversity

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(population, func)
        evaluations = self.population_size
        best_index = np.argmin(fitness)

        while evaluations < self.budget:
            diversity_measure = self.calculate_diversity(population)
            mutants = self.mutate(population, best_index, diversity_measure)
            trials = np.array(
                [
                    self.crossover(population[i], mutants[i], diversity_measure)
                    for i in range(self.population_size)
                ]
            )
            fitness_trials = self.evaluate(trials, func)
            evaluations += len(trials)

            # Selection with elitism
            combined_population = np.vstack((population, trials))
            combined_fitness = np.hstack((fitness, fitness_trials))
            sorted_indices = np.argsort(combined_fitness)[: self.population_size]
            population = combined_population[sorted_indices]
            fitness = combined_fitness[sorted_indices]

            best_index = np.argmin(fitness)

        return fitness[best_index], population[best_index]
