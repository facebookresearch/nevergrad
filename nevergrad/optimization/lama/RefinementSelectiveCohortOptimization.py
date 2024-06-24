import numpy as np


class RefinementSelectiveCohortOptimization:
    def __init__(self, budget, dimension=5, population_size=100, elite_fraction=0.1, mutation_scale=0.05):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(population_size * elite_fraction)
        self.mutation_scale = mutation_scale

    def __call__(self, func):
        # Initialize population within the bounds [-5, 5]
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(x) for x in population])
        evaluations = self.population_size

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        while evaluations < self.budget:
            new_population = np.empty_like(population)
            elite_indices = np.argsort(fitness)[: self.elite_count]
            elite_population = population[elite_indices]

            # Calculate weighted probabilities for selection based on inverse fitness rank
            ranks = np.argsort(np.argsort(fitness))
            selection_probabilities = (self.population_size - ranks) / np.sum(self.population_size - ranks)

            for i in range(self.population_size):
                # Select parents based on fitness-proportional selection
                parents_indices = np.random.choice(
                    self.population_size, 2, p=selection_probabilities, replace=False
                )
                parent1, parent2 = population[parents_indices]

                # Perform crossover and mutation
                crossover_point = np.random.randint(1, self.dimension)
                child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                mutation_vector = self.mutation_scale * np.random.randn(self.dimension)
                child += mutation_vector

                # Ensure child stays within bounds
                child = np.clip(child, -5.0, 5.0)
                child_fitness = func(child)
                evaluations += 1

                new_population[i] = child

                if child_fitness < best_fitness:
                    best_fitness = child_fitness
                    best_individual = child

                if evaluations >= self.budget:
                    break

            population = new_population
            fitness = np.array([func(x) for x in population])

            # Adapt mutation scale by reducing it
            self.mutation_scale *= 0.99

        return best_fitness, best_individual
