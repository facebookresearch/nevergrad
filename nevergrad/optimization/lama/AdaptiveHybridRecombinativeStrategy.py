import numpy as np


class AdaptiveHybridRecombinativeStrategy:
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

    def adapt_mutation_rate(self, best_fitness, current_fitness):
        improvement = best_fitness - current_fitness
        if improvement > 0:
            return max(0.01, 1 - np.log1p(improvement))
        else:
            return 0.1

    def mutate_population(self, population, mutation_rate):
        mutations = np.random.normal(0, mutation_rate, population.shape)
        return np.clip(population + mutations, self.lower_bound, self.upper_bound)

    def recombine_population(self, best_individuals, population_size):
        num_top = best_individuals.shape[0]
        choices = np.random.choice(num_top, size=population_size)
        mix_ratio = np.random.beta(
            2, 5, size=(population_size, self.dim)
        )  # Ensuring dimensionality is maintained
        recombined_population = (
            mix_ratio * best_individuals[choices] + (1 - mix_ratio) * best_individuals[choices[::-1]]
        )
        return recombined_population

    def __call__(self, func):
        population_size = 100
        num_best = 10  # Elite group size

        population = self.generate_initial_population(population_size)
        best_score = float("inf")
        best_individual = None

        while self.budget > 0:
            fitness = self.evaluate_population(func, population)
            self.budget -= population_size  # Reducing the remaining budget

            best_individuals, best_fitness = self.select_top_individuals(population, fitness, num_best)

            if best_fitness[0] < best_score:
                best_score = best_fitness[0]
                best_individual = best_individuals[0]

            mutation_rate = self.adapt_mutation_rate(best_score, best_fitness[0])
            new_population = self.recombine_population(best_individuals, population_size)
            population = self.mutate_population(new_population, mutation_rate)

        return best_score, best_individual
