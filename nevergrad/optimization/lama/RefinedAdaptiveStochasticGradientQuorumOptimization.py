import numpy as np


class RefinedAdaptiveStochasticGradientQuorumOptimization:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=100,
        elite_fraction=0.3,
        mutation_scale=0.1,
        momentum=0.5,
        learning_rate=0.05,
        decay_rate=0.99,
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(max(1, population_size * elite_fraction))
        self.mutation_scale = mutation_scale
        self.momentum = momentum
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate  # Adaptive decay rate for learning rate

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.population_size

        # Track the best solution
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]
        velocity = np.zeros(self.dimension)

        while evaluations < self.budget:
            new_population = np.empty_like(population)
            for i in range(self.population_size):
                # Select elite indices including the best individual
                elite_indices = np.random.choice(self.population_size, self.elite_count - 1, replace=False)
                elite_indices = np.append(elite_indices, best_idx)
                elite_individuals = population[elite_indices]
                elite_fitness = fitness[elite_indices]

                # Determine the local best among the elites
                local_best_idx = np.argmin(elite_fitness)
                local_best = elite_individuals[local_best_idx]

                # Modified update strategy using a weighted gradient and momentum
                gradient = best_individual - local_best
                random_noise = np.random.normal(0, self.mutation_scale, self.dimension)
                mutation = (gradient * random_noise + self.momentum * velocity) * self.learning_rate
                child = np.clip(local_best + mutation, -5.0, 5.0)
                child_fitness = func(child)
                evaluations += 1

                # Update the best solution if necessary
                if child_fitness < best_fitness:
                    best_fitness = child_fitness
                    best_individual = child
                    velocity = child - local_best

                new_population[i, :] = child

                if evaluations >= self.budget:
                    break

            population = new_population
            fitness = np.array([func(ind) for ind in population])

            # Adaptive parameters modifications
            self.learning_rate *= self.decay_rate
            self.mutation_scale *= np.random.uniform(0.9, 1.1)
            self.elite_count = int(max(1, self.population_size * np.random.uniform(0.25, 0.35)))

        return best_fitness, best_individual
