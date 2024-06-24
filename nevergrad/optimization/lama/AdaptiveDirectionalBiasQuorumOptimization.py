import numpy as np


class AdaptiveDirectionalBiasQuorumOptimization:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=100,
        elite_fraction=0.2,
        mutation_scale=0.25,
        momentum=0.9,
        adaptive_rate=0.1,
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = max(1, int(population_size * elite_fraction))
        self.mutation_scale = mutation_scale
        self.momentum = momentum
        self.adaptive_rate = adaptive_rate

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(individual) for individual in population])
        evaluations = self.population_size

        # Initialize best solution tracking
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]
        velocity = np.zeros(self.dimension)

        # Main optimization loop
        while evaluations < self.budget:
            new_population = []
            for i in range(self.population_size):
                # Select a quorum including the best individual
                quorum_indices = np.random.choice(self.population_size, self.elite_count - 1, replace=False)
                quorum_indices = np.append(quorum_indices, best_idx)
                quorum = population[quorum_indices]
                quorum_fitness = fitness[quorum_indices]

                # Determine the local best
                local_best_idx = np.argmin(quorum_fitness)
                local_best = quorum[local_best_idx]

                # Adaptive mutation based on local best, global best, and velocity
                direction = best_individual - local_best
                random_direction = np.random.normal(0, self.mutation_scale, self.dimension)
                mutation = random_direction * direction + self.momentum * velocity
                child = np.clip(local_best + mutation, -5.0, 5.0)
                child_fitness = func(child)
                evaluations += 1

                # Update the best solution and velocity
                if child_fitness < best_fitness:
                    velocity = self.momentum * velocity + self.adaptive_rate * (child - best_individual)
                    best_fitness = child_fitness
                    best_individual = child

                new_population.append(child)

                if evaluations >= self.budget:
                    break

            population = np.array(new_population)
            fitness = np.array([func(ind) for ind in population])

            # Adaptively update mutation scale and elite count
            self.mutation_scale *= 1 + self.adaptive_rate * np.random.uniform(-1, 1)
            self.elite_count = max(
                1, int(self.elite_count * (1 + self.adaptive_rate * np.random.uniform(-0.1, 0.1)))
            )

        return best_fitness, best_individual
