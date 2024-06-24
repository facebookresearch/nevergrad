import numpy as np


class AdaptiveGuidedEvolutionStrategy:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=50,
        step_size=0.1,
        decay_rate=0.995,
    ):
        self.budget = budget
        self.dimension = dimension
        self.bounds = np.array([lower_bound, upper_bound])
        self.population_size = population_size
        self.step_size = step_size  # Initial step size for mutation
        self.decay_rate = decay_rate  # Decay rate for step size to reduce it each generation

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def mutate(self, individual, gen):
        # Mutation using adaptive step size
        mutation_strength = self.step_size * (self.decay_rate**gen)  # Decaying step size
        mutation = np.random.normal(0, mutation_strength, self.dimension)
        return np.clip(individual + mutation, self.bounds[0], self.bounds[1])

    def select_best(self, population, fitness):
        # Tournament selection for simplicity
        best_index = np.argmin(fitness)
        return population[best_index], fitness[best_index]

    def __call__(self, func):
        population = self.initialize_population()
        fitness = np.array([func(ind) for ind in population])
        best_individual, best_fitness = self.select_best(population, fitness)

        evaluations = self.population_size

        for gen in range(1, self.budget // self.population_size):
            new_population = np.array([self.mutate(ind, gen) for ind in population])
            new_fitness = np.array([func(ind) for ind in new_population])
            evaluations += self.population_size

            for i in range(self.population_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
                    if new_fitness[i] < best_fitness:
                        best_fitness = new_fitness[i]
                        best_individual = new_population[i]

            # Update the step size adaptively
            if evaluations >= self.budget:
                break

        return best_fitness, best_individual
