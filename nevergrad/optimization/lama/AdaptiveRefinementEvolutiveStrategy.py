import numpy as np


class AdaptiveRefinementEvolutiveStrategy:
    def __init__(self, budget, dim=5):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def generate_initial_population(self, size=100):
        return np.random.uniform(self.lower_bound, self.upper_bound, (size, self.dim))

    def evaluate_population(self, func, population):
        return np.array([func(ind) for ind in population])

    def select_best(self, population, fitness, num_select):
        indices = np.argsort(fitness)[:num_select]
        return population[indices], fitness[indices]

    def mutate(self, population, mutation_rate=0.05, mutation_strength=0.5):
        mutation_mask = np.random.rand(*population.shape) < mutation_rate
        mutation_values = np.random.normal(0, mutation_strength, population.shape)
        new_population = population + mutation_mask * mutation_values
        return np.clip(new_population, self.lower_bound, self.upper_bound)

    def crossover(self, parents, crossover_rate=0.95):
        new_population = []
        for _ in range(len(parents)):
            if np.random.rand() < crossover_rate:
                p1, p2 = np.random.choice(len(parents), 2, replace=False)
                alpha = np.random.rand()
                child = alpha * parents[p1] + (1 - alpha) * parents[p2]
            else:
                child = parents[np.random.randint(len(parents))]
            new_population.append(child)
        return np.array(new_population)

    def __call__(self, func):
        # Parameters
        population_size = 100
        num_generations = self.budget // population_size
        mutation_rate = 0.05
        mutation_strength = 0.5
        crossover_rate = 0.95

        # Initialize
        population = self.generate_initial_population(population_size)
        best_score = float("inf")
        best_individual = None

        # Evolution loop
        for gen in range(num_generations):
            fitness = self.evaluate_population(func, population)
            best_population, best_fitness = self.select_best(population, fitness, population_size // 5)

            if best_fitness[0] < best_score:
                best_score = best_fitness[0]
                best_individual = best_population[0]

            # Generate new population using crossover and mutation
            population = self.crossover(best_population, crossover_rate)
            population = self.mutate(population, mutation_rate, mutation_strength)

            # Adaptive mutation adjustments
            if gen % 10 == 0 and gen > 0:
                mutation_rate /= 1.1  # Decrease mutation rate slowly
                mutation_strength /= 1.1  # Decrease mutation strength to fine-tune exploration

        return best_score, best_individual
