import numpy as np


class OptimizedEvolutiveStrategy:
    def __init__(self, budget, dim=5):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def generate_initial_population(self, size=10):
        return np.random.uniform(self.lower_bound, self.upper_bound, (size, self.dim))

    def evaluate_population(self, func, population):
        return np.array([func(ind) for ind in population])

    def select_best(self, population, fitness, num_best):
        indices = np.argsort(fitness)[:num_best]
        return population[indices], fitness[indices]

    def mutate(self, population, mutation_rate=0.1, mutation_strength=0.5):
        mutation_mask = np.random.rand(*population.shape) < mutation_rate
        mutation_values = np.random.normal(0, mutation_strength, population.shape)
        new_population = population + mutation_mask * mutation_values
        return np.clip(new_population, self.lower_bound, self.upper_bound)

    def __call__(self, func):
        # Parameters
        population_size = 10
        num_generations = self.budget // population_size
        num_best = 2
        mutation_rate = 0.1
        mutation_strength_initial = 0.5
        decay_factor = 0.99

        # Initialize
        population = self.generate_initial_population(population_size)
        best_score = float("inf")
        best_individual = None

        # Evolution loop
        for _ in range(num_generations):
            fitness = self.evaluate_population(func, population)
            best_population, best_fitness = self.select_best(population, fitness, num_best)

            if best_fitness[0] < best_score:
                best_score = best_fitness[0]
                best_individual = best_population[0]

            # Generate new population
            population = self.mutate(best_population, mutation_rate, mutation_strength_initial)
            mutation_strength_initial *= decay_factor

        return best_score, best_individual
