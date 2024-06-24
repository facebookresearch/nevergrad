import numpy as np


class EnhancedOptimizedEvolutiveStrategy:
    def __init__(self, budget, dim=5):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def generate_initial_population(self, size=20):
        return np.random.uniform(self.lower_bound, self.upper_bound, (size, self.dim))

    def evaluate_population(self, func, population):
        return np.array([func(ind) for ind in population])

    def select_best(self, population, fitness, num_best):
        indices = np.argsort(fitness)[:num_best]
        return population[indices], fitness[indices]

    def mutate(self, population, mutation_rate=0.2, mutation_strength=0.8):
        mutation_mask = np.random.rand(*population.shape) < mutation_rate
        mutation_values = np.random.normal(0, mutation_strength, population.shape)
        new_population = population + mutation_mask * mutation_values
        return np.clip(new_population, self.lower_bound, self.upper_bound)

    def crossover(self, parent1, parent2):
        alpha = np.random.rand(self.dim)
        return alpha * parent1 + (1 - alpha) * parent2

    def __call__(self, func):
        # Parameters
        population_size = 20
        num_generations = self.budget // population_size
        num_best = 4
        mutation_rate = 0.2
        mutation_strength_initial = 0.8
        decay_factor = 0.95

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
            new_population = []
            while len(new_population) < population_size:
                for i in range(num_best):
                    for j in range(i + 1, num_best):
                        child = self.crossover(best_population[i], best_population[j])
                        new_population.append(child)
                        if len(new_population) >= population_size:
                            break
            population = np.array(new_population)
            population = self.mutate(population, mutation_rate, mutation_strength_initial)
            mutation_strength_initial *= decay_factor

        return best_score, best_individual
