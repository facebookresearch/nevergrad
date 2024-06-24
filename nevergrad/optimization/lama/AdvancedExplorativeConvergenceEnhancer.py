import numpy as np


class AdvancedExplorativeConvergenceEnhancer:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=100,
        elite_fraction=0.05,
        mutation_intensity=0.3,
        crossover_probability=0.7,
        elite_boosting_factor=1.1,
        mutation_decay=0.98,
        stabilization_period=10,
    ):
        self.budget = budget
        self.dimension = dimension
        self.lower_bound = np.full(dimension, lower_bound)
        self.upper_bound = np.full(dimension, upper_bound)
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.mutation_intensity = mutation_intensity
        self.crossover_probability = crossover_probability
        self.elite_boosting_factor = elite_boosting_factor
        self.mutation_decay = mutation_decay
        self.stabilization_period = stabilization_period

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dimension))

    def evaluate_fitness(self, func, population):
        return np.array([func(individual) for individual in population])

    def select_elites(self, population, fitness):
        elite_count = int(self.population_size * self.elite_fraction)
        elite_indices = np.argsort(fitness)[:elite_count]
        return population[elite_indices], fitness[elite_indices]

    def mutate(self, individual):
        mutation = np.random.normal(0, self.mutation_intensity, self.dimension)
        return np.clip(individual + mutation, self.lower_bound, self.upper_bound)

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_probability:
            alpha = np.random.rand(self.dimension)
            return np.clip(alpha * parent1 + (1 - alpha) * parent2, self.lower_bound, self.upper_bound)
        return np.copy(parent1 if np.random.rand() < 0.5 else parent2)

    def elite_boosting(self, elite_individual):
        perturbation = np.random.normal(
            0, self.mutation_intensity * self.elite_boosting_factor, self.dimension
        )
        return np.clip(elite_individual + perturbation, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate_fitness(func, population)
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]
        evaluations = self.population_size

        stabilization_counter = 0
        while evaluations < self.budget:
            elites, elite_fitness = self.select_elites(population, fitness)
            new_population = np.array([self.elite_boosting(elite) for elite in elites])

            while len(new_population) < self.population_size:
                parents_indices = np.random.choice(len(elites), 2, replace=False)
                parent1, parent2 = elites[parents_indices[0]], elites[parents_indices[1]]
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population = np.append(new_population, [child], axis=0)

            population = new_population
            fitness = self.evaluate_fitness(func, population)

            min_idx = np.argmin(fitness)
            min_fitness = fitness[min_idx]

            if min_fitness < best_fitness:
                best_fitness = min_fitness
                best_individual = population[min_idx]
                stabilization_counter = 0
            else:
                stabilization_counter += 1

            evaluations += self.population_size

            if stabilization_counter >= self.stabilization_period:
                self.mutation_intensity *= self.mutation_decay

        return best_fitness, best_individual
