import numpy as np


class RefinedDualConvergenceEvolutiveStrategy:
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

    def mutate(self, population, mutation_rate, mutation_strength):
        mutation_mask = np.random.rand(*population.shape) < mutation_rate
        mutation_values = np.random.normal(0, mutation_strength, population.shape)
        new_population = population + mutation_mask * mutation_values
        return np.clip(new_population, self.lower_bound, self.upper_bound)

    def crossover(self, parents, num_children):
        new_population = []
        for _ in range(num_children):
            if np.random.rand() < 0.95:  # Increased crossover probability
                p1, p2 = np.random.choice(len(parents), 2, replace=False)
                alpha = np.random.rand()
                child = alpha * parents[p1] + (1 - alpha) * parents[p2]
            else:
                child = parents[np.random.randint(len(parents))]
            new_population.append(child)
        return np.array(new_population)

    def __call__(self, func):
        population_size = 250  # Increased initial population size for diversity
        num_generations = self.budget // population_size
        elitism_size = population_size // 5  # Increased elitism to 20%
        mutation_rate = 0.15  # More aggressive initial mutation
        mutation_strength = 1.2  # Higher mutation strength

        population = self.generate_initial_population(population_size)
        best_score = float("inf")
        best_individual = None

        for gen in range(num_generations):
            fitness = self.evaluate_population(func, population)
            best_population, best_fitness = self.select_best(population, fitness, elitism_size)

            if best_fitness[0] < best_score:
                best_score = best_fitness[0]
                best_individual = best_population[0]

            non_elite_size = population_size - elitism_size
            offspring = self.crossover(best_population, non_elite_size)
            offspring = self.mutate(offspring, mutation_rate, mutation_strength)
            population = np.vstack((best_population, offspring))

            # Dynamically adapt mutation rate and strength
            mutation_rate *= 0.95  # Slower rate of mutation decrease
            mutation_strength *= 0.95  # Slower strength decrease

        return best_score, best_individual
