import numpy as np


class RefinedAdaptiveGuidedEvolutionStrategy:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=50,
        initial_step_size=0.5,
        min_step_size=0.01,
    ):
        self.budget = budget
        self.dimension = dimension
        self.bounds = np.array([lower_bound, upper_bound])
        self.population_size = population_size
        self.step_size = initial_step_size
        self.min_step_size = min_step_size

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def mutate(self, individual):
        # Mutation with dynamic step size control
        mutation = np.random.normal(0, self.step_size, self.dimension)
        return np.clip(individual + mutation, self.bounds[0], self.bounds[1])

    def update_step_size(self, generation):
        # Exponential decay with a lower limit
        decay_factor = 0.98
        self.step_size = max(self.min_step_size, self.step_size * decay_factor**generation)

    def evaluate_population(self, func, population):
        return np.array([func(ind) for ind in population])

    def select_best(self, population, fitness):
        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate_population(func, population)
        best_individual, best_fitness = self.select_best(population, fitness)

        evaluations = self.population_size
        generation = 0

        while evaluations < self.budget:
            self.update_step_size(generation)
            new_population = np.array([self.mutate(ind) for ind in population])
            new_fitness = self.evaluate_population(func, new_population)

            # Replace only if the new individual is better
            for i in range(self.population_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
                    if new_fitness[i] < best_fitness:
                        best_fitness = new_fitness[i]
                        best_individual = new_population[i]

            evaluations += self.population_size
            generation += 1

        return best_fitness, best_individual
