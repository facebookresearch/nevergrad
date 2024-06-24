import numpy as np


class RefinedAdaptiveDimensionalCrossoverEvolver:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=100,
        elite_fraction=0.2,
        mutation_intensity=0.05,
        crossover_rate=0.85,
        momentum=0.1,
    ):
        self.budget = budget
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.num_elites = int(population_size * elite_fraction)
        self.mutation_intensity = mutation_intensity
        self.crossover_rate = crossover_rate
        self.momentum = momentum

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dimension))

    def evaluate_fitness(self, func, population):
        return np.array([func(ind) for ind in population])

    def select_elites(self, population, fitness):
        elite_indices = np.argsort(fitness)[: self.num_elites]
        return population[elite_indices], fitness[elite_indices]

    def mutate(self, individual):
        mutation = np.random.normal(0, self.mutation_intensity, self.dimension)
        return np.clip(individual + mutation, self.lower_bound, self.upper_bound)

    def adaptive_crossover(self, parent1, parent2):
        crossover_mask = np.random.rand(self.dimension) < self.crossover_rate
        child = np.where(crossover_mask, parent1, parent2)
        return child

    def momentum_update(self, current, previous):
        return current + self.momentum * (current - previous)

    def reproduce(self, elites, elite_fitness, previous_population=None):
        new_population = np.empty((self.population_size, self.dimension))
        previous_best = elites[np.argmin(elite_fitness)]

        for i in range(self.population_size):
            parents = np.random.choice(self.num_elites, 2, replace=False)
            child = self.adaptive_crossover(elites[parents[0]], elites[parents[1]])
            child = self.mutate(child)
            if previous_population is not None:
                child = self.momentum_update(child, previous_population[i])
            new_population[i] = child

        # Introduce the best of the previous generation to the new population
        new_population[0] = previous_best
        return new_population

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate_fitness(func, population)
        previous_population = None

        best_individual = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        evaluations = self.population_size

        while evaluations < self.budget:
            elites, elite_fitness = self.select_elites(population, fitness)
            population = self.reproduce(elites, elite_fitness, previous_population)
            fitness = self.evaluate_fitness(func, population)

            current_best_fitness = np.min(fitness)
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[np.argmin(fitness)]

            previous_population = population
            evaluations += self.population_size

        return best_fitness, best_individual
