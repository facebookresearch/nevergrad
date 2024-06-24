import numpy as np


class StochasticGradientHybridOptimization:
    def __init__(self, budget, dim=5):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def generate_initial_population(self, size=100):
        return np.random.uniform(self.lower_bound, self.upper_bound, (size, self.dim))

    def evaluate_population(self, func, population):
        return np.array([func(ind) for ind in population])

    def select_top_individuals(self, population, fitness, num_best):
        indices = np.argsort(fitness)[:num_best]
        return population[indices], fitness[indices]

    def adapt_mutation_strength(self, generation, base_strength=1.0, decay_rate=0.98):
        return base_strength * (decay_rate**generation)

    def mutate_population(self, population, strength):
        mutations = np.random.normal(0, strength, population.shape)
        return np.clip(population + mutations, self.lower_bound, self.upper_bound)

    def hybridize(self, best_individuals, mutation_strength, population_size):
        num_top = len(best_individuals)
        new_population = np.tile(best_individuals, (population_size // num_top, 1))
        return self.mutate_population(new_population, mutation_strength)

    def __call__(self, func):
        population_size = 200
        num_generations = self.budget // population_size
        num_best = 10  # Top individuals to focus on

        population = self.generate_initial_population(population_size)
        best_score = float("inf")
        best_individual = None

        for gen in range(num_generations):
            fitness = self.evaluate_population(func, population)
            best_individuals, best_fitness = self.select_top_individuals(population, fitness, num_best)

            if best_fitness[0] < best_score:
                best_score = best_fitness[0]
                best_individual = best_individuals[0]

            # Adaptively change mutation strength
            strength = self.adapt_mutation_strength(gen)
            population = self.hybridize(best_individuals, strength, population_size)

        return best_score, best_individual
