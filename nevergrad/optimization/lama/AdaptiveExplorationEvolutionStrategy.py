import numpy as np


class AdaptiveExplorationEvolutionStrategy:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=100,
        elite_fraction=0.2,
        mutation_rate_initial=0.3,
        mutation_decrease=0.99,
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(population_size * elite_fraction)
        self.mutation_rate = mutation_rate_initial
        self.mutation_decrease = mutation_decrease
        self.mutation_rate_initial = mutation_rate_initial

    def __call__(self, func):
        # Initialize population within given bounds
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(individual) for individual in population])
        evaluations = self.population_size

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        while evaluations < self.budget:
            # Select elite individuals
            elite_indices = np.argsort(fitness)[: self.elite_count]
            elites = population[elite_indices]

            # Generate offspring from elites with mutation and crossover
            new_population = np.empty((self.population_size, self.dimension))
            for i in range(self.population_size):
                parent1, parent2 = elites[np.random.choice(self.elite_count, 2, replace=False)]
                crossover_point = np.random.randint(self.dimension)
                child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
                mutation = np.random.normal(0, self.mutation_rate, self.dimension)
                child = np.clip(child + mutation, -5.0, 5.0)

                # Evaluate the child
                child_fitness = func(child)
                evaluations += 1

                # Store child
                new_population[i] = child
                if child_fitness < best_fitness:
                    best_fitness = child_fitness
                    best_individual = child

                if evaluations >= self.budget:
                    break

            population = new_population
            fitness = np.array([func(ind) for ind in population])

            # Adapt mutation rate
            if evaluations < self.budget / 2:
                self.mutation_rate *= self.mutation_decrease
            else:
                # Increase mutation rate later in the search to escape local optima
                self.mutation_rate = min(
                    self.mutation_rate_initial, self.mutation_rate / self.mutation_decrease
                )

        return best_fitness, best_individual
