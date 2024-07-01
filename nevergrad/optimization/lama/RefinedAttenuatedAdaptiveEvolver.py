import numpy as np


class RefinedAttenuatedAdaptiveEvolver:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=120,
        initial_step_size=0.3,
        step_decay=0.95,
        elite_ratio=0.25,
        mutation_probability=0.1,
    ):
        self.budget = budget
        self.dimension = dimension
        self.bounds = np.array([lower_bound, upper_bound])
        self.population_size = population_size
        self.step_size = initial_step_size
        self.step_decay = step_decay
        self.elite_count = int(population_size * elite_ratio)
        self.mutation_probability = mutation_probability

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def mutate(self, individual, scale):
        if np.random.rand() < self.mutation_probability:
            mutation = np.random.normal(0, scale, self.dimension)
            return np.clip(individual + mutation, self.bounds[0], self.bounds[1])
        return individual

    def evaluate_population(self, func, population):
        return np.array([func(ind) for ind in population])

    def select_elites(self, population, fitness):
        elite_indices = np.argsort(fitness)[: self.elite_count]
        return population[elite_indices], fitness[elite_indices]

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate_population(func, population)
        best_individual = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)

        evaluations = self.population_size
        generation = 0

        while evaluations < self.budget:
            scale = self.step_size * (
                self.step_decay**generation
            )  # Adjusted step size decay for better exploration

            new_population = np.array([self.mutate(ind, scale) for ind in population])
            new_fitness = self.evaluate_population(func, new_population)

            combined_population = np.vstack((population, new_population))
            combined_fitness = np.hstack((fitness, new_fitness))
            indices = np.argsort(combined_fitness)
            population = combined_population[indices[: self.population_size]]
            fitness = combined_fitness[indices[: self.population_size]]

            current_best = population[np.argmin(fitness)]
            current_best_fitness = np.min(fitness)
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best

            evaluations += self.population_size
            generation += 1

        return best_fitness, best_individual
