import numpy as np


class AdaptiveElitistPopulationStrategy:
    def __init__(self, budget, dimension=5, population_size=50, elite_fraction=0.2, mutation_intensity=0.05):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(population_size * elite_fraction)
        self.mutation_intensity = mutation_intensity  # Mutation intensity factor

    def __call__(self, func):
        # Initialize the population within bounds [-5, 5]
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(individual) for individual in population])
        evaluations = self.population_size

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        while evaluations < self.budget:
            # Identify elite individuals
            elite_indices = np.argsort(fitness)[: self.elite_count]
            elites = population[elite_indices]

            # Generate new population
            new_population = np.empty_like(population)
            for i in range(self.population_size):
                if i < self.elite_count:
                    # Elite individuals are carried over unchanged
                    new_population[i] = population[elite_indices[i]]
                else:
                    # Generate new individuals by mutating elite individuals
                    elite = elites[np.random.randint(0, self.elite_count)]
                    mutation = np.random.normal(0, self.adaptive_mutation_scale(evaluations), self.dimension)
                    new_individual = np.clip(elite + mutation, -5.0, 5.0)
                    new_population[i] = new_individual

                # Evaluate new individual's fitness
                new_fitness = func(new_population[i])
                evaluations += 1

                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_individual = new_population[i]

                if evaluations >= self.budget:
                    break

            population = new_population
            fitness = np.array([func(individual) for individual in population])

        return best_fitness, best_individual

    def adaptive_mutation_scale(self, evaluations):
        # Decrease mutation scale as the number of evaluations increases
        return self.mutation_intensity * (1 - (evaluations / self.budget))
