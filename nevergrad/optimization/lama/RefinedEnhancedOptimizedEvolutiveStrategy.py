import numpy as np


class RefinedEnhancedOptimizedEvolutiveStrategy:
    def __init__(self, budget, dim=5):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def generate_initial_population(self, size=50):
        return np.random.uniform(self.lower_bound, self.upper_bound, (size, self.dim))

    def evaluate_population(self, func, population):
        return np.array([func(ind) for ind in population])

    def select_best(self, population, fitness, num_best):
        indices = np.argsort(fitness)[:num_best]
        return population[indices], fitness[indices]

    def mutate(self, population, mutation_rate=0.1, mutation_strength=1.0):
        mutation_mask = np.random.rand(*population.shape) < mutation_rate
        mutation_values = np.random.normal(0, mutation_strength, population.shape)
        new_population = population + mutation_mask * mutation_values
        return np.clip(new_population, self.lower_bound, self.upper_bound)

    def crossover(self, parent1, parent2, crossover_rate=0.9):
        if np.random.rand() < crossover_rate:
            alpha = np.random.rand(self.dim)
            return alpha * parent1 + (1 - alpha) * parent2
        else:
            return parent1 if np.random.rand() > 0.5 else parent2

    def __call__(self, func):
        # Parameters
        population_size = 50
        num_generations = self.budget // population_size
        num_best = 5
        mutation_rate = 0.1
        mutation_strength = 1.0
        crossover_rate = 0.9
        decay_factor = 0.98

        # Initialize
        population = self.generate_initial_population(population_size)
        best_score = float("inf")
        best_individual = None

        # Evolution loop
        for gen in range(num_generations):
            fitness = self.evaluate_population(func, population)
            best_population, best_fitness = self.select_best(population, fitness, num_best)

            if best_fitness[0] < best_score:
                best_score = best_fitness[0]
                best_individual = best_population[0]

            # Generate new population using crossover and mutation
            new_population = []
            while len(new_population) < population_size:
                parents = np.random.choice(num_best, 2, replace=False)
                child = self.crossover(
                    best_population[parents[0]], best_population[parents[1]], crossover_rate
                )
                new_population.append(child)
            population = np.array(new_population)
            population = self.mutate(population, mutation_rate, mutation_strength)
            mutation_strength *= decay_factor

        return best_score, best_individual
