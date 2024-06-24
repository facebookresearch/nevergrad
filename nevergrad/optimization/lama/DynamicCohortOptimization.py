import numpy as np


class DynamicCohortOptimization:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=50,
        elite_fraction=0.2,
        mutation_scale=0.2,
        learning_rate=0.1,
        learning_rate_decay=0.98,
        mutation_decay=0.99,
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(population_size * elite_fraction)
        self.mutation_scale = mutation_scale
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.mutation_decay = mutation_decay

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(x) for x in population])
        evaluations = self.population_size

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        while evaluations < self.budget:
            new_population = np.zeros_like(population)
            elite_indices = fitness.argsort()[: self.elite_count]
            for i in range(self.population_size):
                # Tournament selection for parent selection
                indices = np.random.choice(elite_indices, 2, replace=False)
                if fitness[indices[0]] < fitness[indices[1]]:
                    parent = population[indices[0]]
                else:
                    parent = population[indices[1]]

                # Mutation based on normal distribution
                mutation = np.random.normal(0, self.mutation_scale, self.dimension)
                child = np.clip(parent + mutation, -5.0, 5.0)
                child_fitness = func(child)
                evaluations += 1

                # Update best solution found
                if child_fitness < best_fitness:
                    best_fitness = child_fitness
                    best_individual = child

                new_population[i] = child

                if evaluations >= self.budget:
                    break

            population = new_population
            fitness = np.array([func(x) for x in population])

            # Adjust learning rate and mutation scale
            self.mutation_scale *= self.mutation_decay
            self.learning_rate *= self.learning_rate_decay

        return best_fitness, best_individual
