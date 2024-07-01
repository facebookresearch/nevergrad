import numpy as np


class ImprovedIterativeAdaptiveGradientEvolver:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=100,
        elite_fraction=0.2,
        mutation_intensity=0.1,
        crossover_probability=0.7,
        gradient_step=0.05,
        mutation_decay=0.98,
        gradient_enhancement=True,
    ):
        self.budget = budget
        self.dimension = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.mutation_intensity = mutation_intensity
        self.crossover_probability = crossover_probability
        self.gradient_step = gradient_step
        self.mutation_decay = mutation_decay
        self.gradient_enhancement = gradient_enhancement

    def initialize_population(self):
        return np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dimension))

    def evaluate_fitness(self, func, population):
        return np.array([func(individual) for individual in population])

    def select_elites(self, population, fitness):
        elite_indices = np.argsort(fitness)[: int(self.population_size * self.elite_fraction)]
        return population[elite_indices], fitness[elite_indices]

    def mutate(self, individual):
        if self.mutation_intensity > 0:
            mutation = np.random.normal(0, self.mutation_intensity, self.dimension)
            return np.clip(individual + mutation, self.lower_bound, self.upper_bound)
        return individual

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_probability:
            alpha = np.random.rand(self.dimension)
            return np.clip(alpha * parent1 + (1 - alpha) * parent2, self.lower_bound, self.upper_bound)
        return parent1 if np.random.rand() < 0.5 else parent2

    def adaptive_gradient(self, individual, func, best_individual):
        if self.gradient_enhancement:
            gradient_direction = individual - best_individual
            step_size = self.gradient_step / (1 + np.linalg.norm(gradient_direction))
            new_individual = individual - step_size * gradient_direction
            return np.clip(new_individual, self.lower_bound, self.upper_bound)
        return individual

    def __call__(self, func):
        population = self.initialize_population()
        fitness = self.evaluate_fitness(func, population)
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]
        evaluations = self.population_size

        while evaluations < self.budget:
            elites, elite_fitness = self.select_elites(population, fitness)

            new_population = np.zeros_like(population)
            for i in range(self.population_size):
                if i < len(elites):
                    new_population[i] = self.adaptive_gradient(elites[i], func, best_individual)
                else:
                    # Fixing the random choice for Python's zero-dimensional array error.
                    parents_indices = np.random.choice(len(elites), 2, replace=False)
                    parent1 = elites[parents_indices[0]]
                    parent2 = elites[parents_indices[1]]
                    child = self.crossover(parent1, parent2)
                    child = self.mutate(child)
                    new_population[i] = child

            fitness = self.evaluate_fitness(func, new_population)

            min_idx = np.argmin(fitness)
            min_fitness = fitness[min_idx]

            if min_fitness < best_fitness:
                best_fitness = min_fitness
                best_individual = new_population[min_idx]

            population = new_population
            evaluations += self.population_size

            self.mutation_intensity *= self.mutation_decay

        return best_fitness, best_individual
