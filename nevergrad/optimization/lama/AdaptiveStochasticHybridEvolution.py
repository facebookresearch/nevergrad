import numpy as np


class AdaptiveStochasticHybridEvolution:
    def __init__(self, budget, dim=5):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def generate_initial_population(self, size=150):
        return np.random.uniform(self.lower_bound, self.upper_bound, (size, self.dim))

    def evaluate_population(self, func, population):
        return np.array([func(ind) for ind in population])

    def select_top_individuals(self, population, fitness, num_best):
        indices = np.argsort(fitness)[:num_best]
        return population[indices], fitness[indices]

    def adapt_mutation_strength(self, best_score, current_score, base_strength=0.5, scale_factor=0.9):
        if current_score < best_score:
            return base_strength * scale_factor
        else:
            return base_strength / scale_factor

    def mutate_population(self, population, strength):
        mutations = np.random.normal(0, strength, population.shape)
        return np.clip(population + mutations, self.lower_bound, self.upper_bound)

    def recombine_population(self, best_individuals, population_size):
        num_top = len(best_individuals)
        extended_population = np.repeat(best_individuals, population_size // num_top, axis=0)
        random_indices = np.random.randint(0, num_top, size=(population_size, self.dim))
        for i in range(self.dim):
            extended_population[:, i] = best_individuals[random_indices[:, i], i]
        return extended_population

    def __call__(self, func):
        population_size = 150
        num_generations = max(1, self.budget // population_size)
        num_best = 5  # Top individuals to focus on

        population = self.generate_initial_population(population_size)
        best_score = float("inf")
        best_individual = None

        for gen in range(num_generations):
            fitness = self.evaluate_population(func, population)
            best_individuals, best_fitness = self.select_top_individuals(population, fitness, num_best)

            if best_fitness[0] < best_score:
                best_score = best_fitness[0]
                best_individual = best_individuals[0]

            strength = self.adapt_mutation_strength(best_score, best_fitness[0])
            new_population = self.recombine_population(best_individuals, population_size)
            population = self.mutate_population(new_population, strength)

        return best_score, best_individual
