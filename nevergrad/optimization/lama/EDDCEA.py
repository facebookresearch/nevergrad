import numpy as np


class EDDCEA:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 100
        self.elite_size = 10  # Slightly larger elite group for better exploitation

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutate(self, population, best_index, diversity_measure):
        F = 0.8 * (2 - diversity_measure)  # Adjusted mutation factor
        new_population = np.empty_like(population)
        for i in range(self.population_size):
            idxs = np.random.choice(np.delete(np.arange(self.population_size), best_index), 3, replace=False)
            a, b, c = population[idxs]
            mutant_vector = a + F * (b - c)
            new_population[i] = np.clip(mutant_vector, self.bounds[0], self.bounds[1])
        return new_population

    def crossover(self, target, mutant, diversity_measure):
        CR = 0.4 + 0.6 * diversity_measure  # Adjusted crossover probability
        mask = np.random.rand(self.dimension) < CR
        return np.where(mask, mutant, target)

    def calculate_diversity(self, population):
        mean_population = np.mean(population, axis=0)
        diversity = np.mean(np.sqrt(np.sum((population - mean_population) ** 2, axis=1)))
        return diversity / np.sqrt(self.dimension * (self.bounds[1] - self.bounds[0]) ** 2)

    def local_search(self, best_individual, func):
        # Simple local search around the best individual
        perturbation = np.random.normal(0, 0.1, self.dimension)
        new_individual = np.clip(best_individual + perturbation, self.bounds[0], self.bounds[1])
        if func(new_individual) < func(best_individual):
            return new_individual
        return best_individual

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

            # Selection with elitism incorporating local search
            combined_population = np.vstack((population, trials))
            combined_fitness = np.hstack((fitness, fitness_trials))
            sorted_indices = np.argsort(combined_fitness)[: self.population_size]
            population = combined_population[sorted_indices]
            fitness = combined_fitness[sorted_indices]

            best_index = np.argmin(fitness)
            population[best_index] = self.local_search(
                population[best_index], func
            )  # Apply local search on the best

        return fitness[best_index], population[best_index]
