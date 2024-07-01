import numpy as np


class DPADE:
    def __init__(self, budget, population_size=30):
        self.budget = budget
        self.population_size = population_size
        self.dimension = 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dimension))

    def evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])

    def adaptive_parameters(self, iteration, max_iter):
        # Adjust F and CR over time
        F = 0.5 + (0.8 - 0.5) * (1 - iteration / max_iter)
        CR = 0.5 + (0.9 - 0.5) * (iteration / max_iter)
        return F, CR

    def mutation(self, population, best_idx, F):
        new_population = np.zeros_like(population)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = np.random.choice(idxs, 3, replace=False)
            mutant = population[a] + F * (population[b] - population[c])
            new_population[i] = np.clip(mutant, self.lower_bound, self.upper_bound)
        return new_population

    def hybrid_mutation(self, population, best_individual, F):
        # Second strategy using best individual
        new_population = np.zeros_like(population)
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            b, c = np.random.choice(idxs, 2, replace=False)
            mutant = best_individual + F * (population[b] - population[c])
            new_population[i] = np.clip(mutant, self.lower_bound, self.upper_bound)
        return new_population

    def crossover(self, population, mutant_population, CR):
        crossover_population = np.array(
            [
                np.where(np.random.rand(self.dimension) < CR, mutant_population[i], population[i])
                for i in range(self.population_size)
            ]
        )
        return crossover_population

    def select(self, population, fitness, trial_population, func):
        trial_fitness = self.evaluate_population(trial_population, func)
        for i in range(self.population_size):
            if trial_fitness[i] < fitness[i]:
                fitness[i] = trial_fitness[i]
                population[i] = trial_population[i]
        return population, fitness

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate_population(population, func)
        best_idx = np.argmin(fitness)

        evaluations = self.population_size
        max_iterations = self.budget // self.population_size

        for iteration in range(max_iterations):
            F, CR = self.adaptive_parameters(iteration, max_iterations)
            mutant_population = self.mutation(population, best_idx, F)
            mutant_population = (
                mutant_population + self.hybrid_mutation(population, population[best_idx], F)
            ) / 2
            crossover_population = self.crossover(population, mutant_population, CR)
            population, fitness = self.select(population, fitness, crossover_population, func)
            evaluations += self.population_size
            best_idx = np.argmin(fitness)

        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]
        return best_fitness, best_solution
