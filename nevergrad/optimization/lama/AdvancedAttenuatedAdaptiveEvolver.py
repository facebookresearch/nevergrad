import numpy as np


class AdvancedAttenuatedAdaptiveEvolver:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=150,
        initial_step_size=0.5,
        step_decay=0.97,
        elite_ratio=0.2,
        mutation_probability=0.2,
        recombination_rate=0.3,
    ):
        self.budget = budget
        self.dimension = dimension
        self.bounds = np.array([lower_bound, upper_bound])
        self.population_size = population_size
        self.step_size = initial_step_size
        self.step_decay = step_decay
        self.elite_count = int(population_size * elite_ratio)
        self.mutation_probability = mutation_probability
        self.recombination_rate = recombination_rate

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def mutate(self, individual, scale):
        if np.random.rand() < self.mutation_probability:
            mutation = np.random.normal(0, scale, self.dimension)
            return np.clip(individual + mutation, self.bounds[0], self.bounds[1])
        return individual

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.recombination_rate:
            crossover_point = np.random.randint(1, self.dimension)
            child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            return child
        return parent1

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
            )  # Dynamic step size for exploration adjustment

            # Pair and possibly crossover individuals for new population
            np.random.shuffle(population)
            new_population = []
            for i in range(0, self.population_size, 2):
                parent1 = population[i]
                parent2 = population[i + 1] if i + 1 < self.population_size else population[0]
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)
                new_population.extend([self.mutate(child1, scale), self.mutate(child2, scale)])
            new_population = np.array(new_population[: self.population_size])
            new_fitness = self.evaluate_population(func, new_population)

            population = np.vstack((population, new_population))
            fitness = np.hstack((fitness, new_fitness))
            indices = np.argsort(fitness)
            population = population[indices[: self.population_size]]
            fitness = fitness[indices[: self.population_size]]

            current_best = population[np.argmin(fitness)]
            current_best_fitness = np.min(fitness)
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = current_best

            evaluations += self.population_size
            generation += 1

        return best_fitness, best_individual
