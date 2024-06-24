import numpy as np


class UltraRefinedAdaptiveConvergenceStrategy:
    def __init__(self, budget, dim=5):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def generate_initial_population(self, size=150):
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
            if np.random.rand() < 0.95:  # High crossover probability
                p1, p2 = np.random.choice(len(parents), 2, replace=False)
                alpha = np.random.rand()
                child = alpha * parents[p1] + (1 - alpha) * parents[p2]
            else:  # Slight chance to pass a direct parent
                child = parents[np.random.randint(len(parents))]
            new_population.append(child)
        return np.array(new_population)

    def __call__(self, func):
        population_size = 300
        num_generations = self.budget // population_size
        elitism_size = int(population_size * 0.3)  # Increased elitism size
        mutation_rate = 0.07  # Reduced mutation rate for initial stability
        mutation_strength = 0.7  # Reduced mutation strength

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

            # Adaptive mutation strategy based on generational feedback
            if gen % 5 == 0 and mutation_rate > 0.01:
                mutation_rate -= 0.005  # Gradual decrease of mutation rate
                mutation_strength *= 0.95  # Gradual decrease of mutation strength

        return best_score, best_individual
