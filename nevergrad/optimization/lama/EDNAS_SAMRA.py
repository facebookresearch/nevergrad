import numpy as np


class EDNAS_SAMRA:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 100
        self.mutation_scale = 0.8  # Initial mutation scale
        self.elite_size = int(self.population_size * 0.1)  # 10% of the population
        self.mutation_adjustment_factor = 0.05  # Rate of mutation scale adjustment

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
        best_fitness = np.min(fitness)
        prev_best_fitness = best_fitness

        while evaluations < self.budget:
            F = np.random.normal(self.mutation_scale, 0.1)  # Use normal distribution for mutation scale
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

            if best_fitness < prev_best_fitness:
                prev_best_fitness = best_fitness
                self.mutation_scale *= 1 - self.mutation_adjustment_factor
            else:
                self.mutation_scale *= 1 + self.mutation_adjustment_factor

            # Check population diversity and introduce random individuals if needed
            if np.std(population.flatten()) < 0.1:
                reinit_indices = np.random.choice(
                    len(population), size=int(self.population_size * 0.1), replace=False
                )
                population[reinit_indices] = np.random.uniform(
                    self.bounds[0], self.bounds[1], (len(reinit_indices), self.dimension)
                )

        return best_fitness, population[np.argmin(fitness)]
