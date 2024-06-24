import numpy as np


class PrecisionTunedEvolver:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=200,
        elite_fraction=0.05,
        adaptive_mutation=True,
    ):
        self.budget = budget
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.num_elites = int(population_size * elite_fraction)
        self.adaptive_mutation = adaptive_mutation

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dimension))

    def evaluate_fitness(self, func, population):
        return np.array([func(ind) for ind in population])

    def select_elites(self, population, fitness):
        elite_indices = np.argsort(fitness)[: self.num_elites]
        return population[elite_indices], fitness[elite_indices]

    def mutate(self, individual):
        mutation_strength = (self.upper_bound - self.lower_bound) / np.sqrt(self.budget)
        mutation = np.random.normal(0, mutation_strength, self.dimension)
        individual = np.clip(individual + mutation, self.lower_bound, self.upper_bound)
        return individual

    def crossover(self, parent1, parent2):
        alpha = np.random.uniform(0.3, 0.7)
        child = alpha * parent1 + (1 - alpha) * parent2
        return child

    def reproduce(self, elites, elite_fitness):
        new_population = np.empty((self.population_size, self.dimension))
        for i in range(self.population_size):
            parents = np.random.choice(self.num_elites, 2, replace=False)
            child = self.crossover(elites[parents[0]], elites[parents[1]])
            if self.adaptive_mutation:
                child = self.mutate(child)
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
            population = self.reproduce(elites, elite_fitness)
            fitness = self.evaluate_fitness(func, population)

            current_best_fitness = np.min(fitness)
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[np.argmin(fitness)]

            evaluations += self.population_size

        return best_fitness, best_individual
