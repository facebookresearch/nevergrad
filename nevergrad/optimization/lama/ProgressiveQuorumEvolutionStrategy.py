import numpy as np


class ProgressiveQuorumEvolutionStrategy:
    def __init__(
        self, budget, dimension=5, population_size=100, elite_fraction=0.1, mutation_scale=0.1, quorum_size=5
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(population_size * elite_fraction)
        self.mutation_scale = mutation_scale
        self.quorum_size = quorum_size

    def __call__(self, func):
        # Initialize population within bounds
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(individual) for individual in population])
        evaluations = self.population_size

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        while evaluations < self.budget:
            # Create new generation with quorum-based selection and mutation
            new_population = np.empty_like(population)
            for i in range(self.population_size):
                # Select quorum randomly and choose the best among them
                quorum_indices = np.random.choice(self.population_size, self.quorum_size, replace=False)
                elite_idx = quorum_indices[np.argmin(fitness[quorum_indices])]
                elite = population[elite_idx]

                # Mutation based on Gaussian noise
                mutation = np.random.normal(0, self.mutation_scale, self.dimension)
                child = np.clip(elite + mutation, -5.0, 5.0)

                # Evaluate new candidate
                child_fitness = func(child)
                evaluations += 1

                # Store the new candidate
                new_population[i] = child
                if child_fitness < best_fitness:
                    best_fitness = child_fitness
                    best_individual = child

                if evaluations >= self.budget:
                    break

            population = new_population
            fitness = np.array([func(ind) for ind in population])

        return best_fitness, best_individual
