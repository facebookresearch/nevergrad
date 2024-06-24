import numpy as np


class EDAEA:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 50
        self.speciation_threshold = 0.5  # Distance threshold for speciation
        self.initial_cr = 0.9
        self.initial_f = 0.8
        self.initial_temp = 1.0
        self.final_temp = 0.01
        self.alpha = 0.95  # Standard cooling rate

    def initialize_population(self):
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))
        # Opposite-based learning initialization
        if np.random.rand() < 0.5:
            means = np.mean(population, axis=0)
            opposite_population = 2 * means - population
            population = np.vstack((population, opposite_population))
            population = population[: self.population_size]  # Ensure population size consistency
        return population

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutate(self, population, best_idx, f, temperature):
        new_population = np.empty_like(population)
        for i in range(len(population)):
            idxs = np.random.choice(self.population_size, 3, replace=False)
            x1, x2, x3 = population[idxs[0]], population[idxs[1]], population[idxs[2]]
            mutant = population[best_idx] + f * temperature * (x1 - x2 + x3 - population[best_idx])
            # Opposite-based mutation
            if np.random.rand() < 0.5:
                means = np.mean(mutant)
                mutant = 2 * means - mutant
            new_population[i] = np.clip(mutant, self.bounds[0], self.bounds[1])
        return new_population

    def crossover(self, target, mutant, cr):
        crossover_mask = np.random.rand(self.dimension) < cr
        return np.where(crossover_mask, mutant, target)

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(population, func)
        evaluations = self.population_size
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_solution = population[best_idx]
        temperature = self.initial_temp
        cr = self.initial_cr
        f = self.initial_f

        while evaluations < self.budget:
            mutated_population = self.mutate(population, best_idx, f, temperature)
            offspring_population = np.array(
                [
                    self.crossover(population[i], mutated_population[i], cr)
                    for i in range(self.population_size)
                ]
            )
            offspring_fitness = self.evaluate(offspring_population, func)
            evaluations += self.population_size

            for i in range(self.population_size):
                if offspring_fitness[i] < fitness[i] or np.random.rand() < np.exp(
                    (fitness[i] - offspring_fitness[i]) / temperature
                ):
                    population[i], fitness[i] = offspring_population[i], offspring_fitness[i]
                    if fitness[i] < best_fitness:
                        best_fitness, best_solution, best_idx = fitness[i], population[i], i

            # Adjust temperature dynamically based on performance
            if evaluations % 100 == 0:
                improvement_rate = (np.min(fitness) - best_fitness) / best_fitness
                if improvement_rate < 0.01:
                    temperature *= self.alpha**2  # Faster cooling if stagnation
                else:
                    temperature *= self.alpha

        return best_fitness, best_solution
