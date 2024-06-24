import numpy as np


class RefinedProgressiveQuorumEvolutionStrategy:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=100,
        elite_fraction=0.1,
        mutation_scale=0.1,
        quorum_size=5,
        adaptive_mutation=True,
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(population_size * elite_fraction)
        self.mutation_scale = mutation_scale
        self.quorum_size = quorum_size
        self.adaptive_mutation = adaptive_mutation

    def __call__(self, func):
        # Initialize population uniformly within bounds
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(individual) for individual in population])
        evaluations = self.population_size

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        # Adaptive mutation scale factor
        scale_factor = self.mutation_scale

        while evaluations < self.budget:
            new_population = np.empty_like(population)
            for i in range(self.population_size):
                # Select a quorum randomly and choose the best among them
                quorum_indices = np.random.choice(self.population_size, self.quorum_size, replace=False)
                elite_idx = quorum_indices[np.argmin(fitness[quorum_indices])]
                elite = population[elite_idx]

                # Mutation based on Gaussian noise
                mutation = np.random.normal(0, scale_factor, self.dimension)
                child = np.clip(elite + mutation, -5.0, 5.0)
                child_fitness = func(child)
                evaluations += 1

                new_population[i] = child
                if child_fitness < best_fitness:
                    best_fitness = child_fitness
                    best_individual = child

                if evaluations >= self.budget:
                    break

            # Update population and fitness
            population = new_population
            fitness = np.array([func(ind) for ind in population])

            # Adaptively adjust mutation scale if enabled
            if self.adaptive_mutation:
                if i % 20 == 0 and scale_factor > 0.01:  # Reduce mutation scale periodically
                    scale_factor *= 0.95

        return best_fitness, best_individual
