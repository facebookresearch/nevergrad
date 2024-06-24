import numpy as np


class RefinedAdaptiveDirectionalBiasQuorumOptimization:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=100,
        elite_fraction=0.2,
        mutation_scale=0.3,
        momentum=0.9,
        learning_rate=0.01,
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(max(1, population_size * elite_fraction))
        self.mutation_scale = mutation_scale
        self.momentum = momentum
        self.learning_rate = learning_rate

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(individual) for individual in population])
        evaluations = self.population_size

        # Track best solution
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]
        velocity = np.zeros(self.dimension)

        # Optimization loop
        while evaluations < self.budget:
            new_population = np.empty_like(population)
            for i in range(self.population_size):
                # Select elite indices including the best individual
                quorum_indices = np.random.choice(self.population_size, self.elite_count - 1, replace=False)
                quorum_indices = np.append(quorum_indices, best_idx)
                quorum = population[quorum_indices]
                quorum_fitness = fitness[quorum_indices]

                # Determine the local best
                local_best_idx = np.argmin(quorum_fitness)
                local_best = quorum[local_best_idx]

                # Mutation and update strategy
                direction = best_individual - local_best
                random_noise = np.random.normal(0, self.mutation_scale, self.dimension)
                mutation = direction * random_noise + self.momentum * velocity
                child = np.clip(local_best + mutation, -5.0, 5.0)
                child_fitness = func(child)
                evaluations += 1

                # Update the best solution and velocity
                if child_fitness < best_fitness:
                    velocity = self.learning_rate * (child - best_individual) + self.momentum * velocity
                    best_fitness = child_fitness
                    best_individual = child

                new_population[i, :] = child

                if evaluations >= self.budget:
                    break

            population = new_population
            fitness = np.array([func(ind) for ind in population])

            # Adapt mutation scale and elite count dynamically
            adaptive_ratio = np.random.uniform(-0.05, 0.05)
            self.mutation_scale *= 1 + adaptive_ratio
            self.elite_count = int(max(1, self.elite_count * (1 + adaptive_ratio)))

        return best_fitness, best_individual
