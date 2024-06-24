import numpy as np


class OptimizedConvergentAdaptiveEvolver:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=150,
        elite_fraction=0.1,
        mutation_rate=0.05,
        mutation_adjustment=0.98,
    ):
        self.budget = budget
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.num_elites = int(population_size * elite_fraction)
        self.mutation_rate = mutation_rate
        self.mutation_adjustment = mutation_adjustment

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dimension))

    def evaluate_fitness(self, func, population):
        return np.array([func(ind) for ind in population])

    def select_elites(self, population, fitness):
        elite_indices = np.argsort(fitness)[: self.num_elites]
        return population[elite_indices], fitness[elite_indices]

    def mutate(self, individual, scale):
        if np.random.rand() < self.mutation_rate:
            mutation = np.random.normal(0, scale, self.dimension)
            return np.clip(individual + mutation, self.lower_bound, self.upper_bound)
        return individual

    def crossover(self, parent1, parent2):
        child = np.zeros(self.dimension)
        for i in range(self.dimension):
            if np.random.rand() > 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        return child

    def reproduce(self, elites, elite_fitness):
        new_population = elites.copy()
        while len(new_population) < self.population_size:
            parents = np.random.choice(self.num_elites, 2, replace=False)
            child = self.crossover(elites[parents[0]], elites[parents[1]])
            child = self.mutate(child, self.mutation_scale)
            new_population = np.vstack([new_population, child])
        return new_population

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate_fitness(func, population)
        self.mutation_scale = (self.upper_bound - self.lower_bound) / 10  # Smaller initial mutation scale

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
            self.mutation_scale *= self.mutation_adjustment  # Gradual decrease in mutation scale

        return best_fitness, best_individual
