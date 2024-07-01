import numpy as np


class EDNAS:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 100
        self.learning_rate = 0.1
        self.mutation_scale = 0.8
        self.elite_size = int(self.population_size * 0.1)  # 10% of the population

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def mutate(self, population, best_idx, F):
        new_population = np.empty_like(population)
        for i in range(len(population)):
            idxs = np.random.choice(np.delete(np.arange(len(population)), best_idx), 3, replace=False)
            x1, x2, x3 = population[idxs]
            mutant_vector = np.clip(x1 + F * (x2 - x3), self.bounds[0], self.bounds[1])
            new_population[i] = mutant_vector
        return new_population

    def crossover(self, target, mutant, CR):
        mask = np.random.rand(self.dimension) < CR
        return np.where(mask, mutant, target)

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(population, func)
        evaluations = self.population_size
        best_index = np.argmin(fitness)
        best_fitness = fitness[best_index]
        previous_best = best_fitness

        while evaluations < self.budget:
            F = np.random.normal(self.mutation_scale, 0.1)  # Adaptive mutation factor
            CR = np.clip(np.std(fitness) / np.ptp(fitness), 0.1, 0.9)  # Adaptive crossover rate
            elite_indices = np.argsort(fitness)[: self.elite_size]
            mutants = self.mutate(population, elite_indices, F)
            trials = np.array(
                [self.crossover(population[i], mutants[i], CR) for i in range(self.population_size)]
            )
            fitness_trials = self.evaluate(trials, func)
            evaluations += self.population_size

            for i in range(self.population_size):
                if fitness_trials[i] < fitness[i]:
                    population[i] = trials[i]
                    fitness[i] = fitness_trials[i]
                    if fitness[i] < best_fitness:
                        best_fitness = fitness[i]
                        best_index = i

            if best_fitness < previous_best:
                self.learning_rate *= 1.1
                previous_best = best_fitness
            else:
                self.learning_rate *= 0.9
                if np.random.rand() < 0.1:
                    reinit_indices = np.random.choice(
                        len(population), size=int(self.population_size * 0.1), replace=False
                    )
                    population[reinit_indices] = np.random.uniform(
                        self.bounds[0], self.bounds[1], (len(reinit_indices), self.dimension)
                    )
                    population[elite_indices] = population[
                        np.argsort(fitness)[: self.elite_size]
                    ]  # Preserve elites

        return best_fitness, population[best_index]
