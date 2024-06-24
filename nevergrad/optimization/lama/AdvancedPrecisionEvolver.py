import numpy as np


class AdvancedPrecisionEvolver:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=100,
        elite_fraction=0.1,
        mutation_factor=0.8,
        crossover_probability=0.7,
    ):
        self.budget = budget
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.num_elites = int(population_size * elite_fraction)
        self.mutation_factor = mutation_factor
        self.crossover_probability = crossover_probability

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dimension))

    def evaluate_fitness(self, func, population):
        return np.array([func(ind) for ind in population])

    def select_elites(self, population, fitness):
        elite_indices = np.argsort(fitness)[: self.num_elites]
        return population[elite_indices], fitness[elite_indices]

    def mutate(self, individual, global_best):
        mutation_strength = np.abs(global_best - individual) * self.mutation_factor
        mutation = np.random.normal(0, mutation_strength)
        new_individual = np.clip(individual + mutation, self.lower_bound, self.upper_bound)
        return new_individual

    def crossover(self, parent1, parent2):
        child = np.where(np.random.rand(self.dimension) < self.crossover_probability, parent1, parent2)
        return child

    def reproduce(self, elites, elite_fitness, global_best):
        new_population = np.empty((self.population_size, self.dimension))
        for i in range(self.population_size):
            parents = np.random.choice(self.num_elites, 2, replace=False)
            child = self.crossover(elites[parents[0]], elites[parents[1]])
            child = self.mutate(child, global_best)
            new_population[i] = child
        return new_population

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate_fitness(func, population)

        best_individual = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        evaluations = self.population_size

        while evaluations < self.budget:
            elites, elite_fitness = self.select_elites(population, fitness)
            global_best = elites[np.argmin(elite_fitness)]
            population = self.reproduce(elites, elite_fitness, global_best)
            fitness = self.evaluate_fitness(func, population)

            current_best_fitness = np.min(fitness)
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[np.argmin(fitness)]

            evaluations += self.population_size

        return best_fitness, best_individual
