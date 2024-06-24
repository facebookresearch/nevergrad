import numpy as np


class AMSEA:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 100
        self.last_best_fitness = np.inf
        self.stagnation_counter = 0

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutate(self, population, best_index, adaptation_factor):
        F = 0.5 + 0.5 * adaptation_factor  # Dynamically adapted mutation factor
        new_population = np.empty_like(population)
        for i in range(self.population_size):
            idxs = np.random.choice(np.delete(np.arange(self.population_size), best_index), 3, replace=False)
            a, b, c = population[idxs]
            mutant_vector = a + F * (b - c)
            new_population[i] = np.clip(mutant_vector, self.bounds[0], self.bounds[1])
        return new_population

    def crossover(self, target, mutant, adaptation_factor):
        CR = 0.2 + 0.6 * adaptation_factor  # Dynamically adapted crossover probability
        mask = np.random.rand(self.dimension) < CR
        return np.where(mask, mutant, target)

    def calculate_adaptation_factor(self, current_best_fitness):
        if current_best_fitness < self.last_best_fitness:
            self.last_best_fitness = current_best_fitness
            self.stagnation_counter = 0
            return 1  # High exploration when progressing
        else:
            self.stagnation_counter += 1
            return max(0, 1 - self.stagnation_counter / 50)  # Increase exploitation if stagnating

    def local_search(self, best_individual, func):
        perturbations = np.random.normal(0, 0.1, (10, self.dimension))
        candidates = np.clip(best_individual + perturbations, self.bounds[0], self.bounds[1])
        fitnesses = self.evaluate(candidates, func)
        best_local_idx = np.argmin(fitnesses)
        return candidates[best_local_idx]

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(population, func)
        evaluations = self.population_size
        best_index = np.argmin(fitness)

        while evaluations < self.budget:
            adaptation_factor = self.calculate_adaptation_factor(fitness[best_index])
            mutants = self.mutate(population, best_index, adaptation_factor)
            trials = np.array(
                [
                    self.crossover(population[i], mutants[i], adaptation_factor)
                    for i in range(self.population_size)
                ]
            )
            fitness_trials = self.evaluate(trials, func)
            evaluations += len(trials)

            combined_population = np.vstack((population, trials))
            combined_fitness = np.hstack((fitness, fitness_trials))
            sorted_indices = np.argsort(combined_fitness)[: self.population_size]
            population = combined_population[sorted_indices]
            fitness = combined_fitness[sorted_indices]

            best_index = np.argmin(fitness)
            population[best_index] = self.local_search(population[best_index], func)

        return fitness[best_index], population[best_index]
