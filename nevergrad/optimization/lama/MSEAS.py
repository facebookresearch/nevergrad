import numpy as np


class MSEAS:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])
        self.population_size = 50
        self.num_subpopulations = 5
        self.subpopulation_size = int(self.population_size / self.num_subpopulations)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.elite_size = 2

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def levy_flight(self):
        return np.random.standard_cauchy(size=self.dimension)

    def adaptive_parameters(self, progress):
        # Adjust mutation factor and crossover rate as the optimization progresses
        self.mutation_factor = 0.8 - progress * 0.5
        self.crossover_rate = 0.7 + progress * 0.2

    def mutate_and_crossover(self, population, func, progress):
        new_population = np.copy(population)
        for i in range(self.population_size):
            if np.random.rand() < 0.1:  # 10% chance of global search mutation
                mutation = self.levy_flight()
            else:
                mutation = self.differential_mutation(population, i)
            mutant = np.clip(population[i] + mutation, self.bounds[0], self.bounds[1])
            child = self.crossover(mutant, population[i])
            new_population[i] = child
        new_fitness = self.evaluate(new_population, func)
        return new_population, new_fitness

    def differential_mutation(self, population, base_idx):
        idxs = np.random.choice(self.population_size, 3, replace=False)
        x1, x2, x3 = population[idxs[0]], population[idxs[1]], population[idxs[2]]
        return self.mutation_factor * (x2 - x3)

    def crossover(self, mutant, target):
        crossover_mask = np.random.rand(self.dimension) < self.crossover_rate
        return np.where(crossover_mask, mutant, target)

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate(population, func)
        evaluations = self.population_size
        best_fitness = np.min(fitness)
        best_solution = population[np.argmin(fitness)]

        while evaluations < self.budget:
            progress = evaluations / self.budget
            self.adaptive_parameters(progress)
            population, fitness = self.mutate_and_crossover(population, func, progress)
            evaluations += self.population_size

            current_best_fitness = np.min(fitness)
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_solution = population[np.argmin(fitness)]

        return best_fitness, best_solution
