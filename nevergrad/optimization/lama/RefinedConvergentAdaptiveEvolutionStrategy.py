import numpy as np


class RefinedConvergentAdaptiveEvolutionStrategy:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=100,
        elite_fraction=0.2,
        mutation_rate=0.1,
        mutation_decrease=0.99,
    ):
        self.budget = budget
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.num_elites = int(population_size * elite_fraction)
        self.mutation_rate = mutation_rate
        self.mutation_decrease = mutation_decrease

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dimension))

    def evaluate_fitness(self, func, population):
        return np.array([func(ind) for ind in population])

    def select_elites(self, population, fitness):
        elite_indices = np.argsort(fitness)[: self.num_elites]
        return population[elite_indices], fitness[elite_indices]

    def mutate(self, individual, scale):
        mutation = np.random.normal(0, scale, self.dimension)
        return np.clip(individual + mutation, self.lower_bound, self.upper_bound)

    def crossover(self, parent1, parent2):
        alpha = np.random.rand()
        return alpha * parent1 + (1 - alpha) * parent2

    def reproduce(self, elites, elite_fitness):
        new_population = elites.copy()
        while len(new_population) < self.population_size:
            parents = np.random.choice(self.num_elites, 2, replace=False)
            child = self.crossover(elites[parents[0]], elites[parents[1]])
            child = self.mutate(child, self.mutation_scale * (np.random.rand() * 2))  # Varying mutation scale
            new_population = np.vstack([new_population, child])
        return new_population

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate_fitness(func, population)
        self.mutation_scale = (self.upper_bound - self.lower_bound) / 2

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

            evaluations += len(population)
            self.mutation_scale *= self.mutation_decrease  # Decrease mutation scale adaptively

        return best_fitness, best_individual
