import numpy as np


class PrecisionAdaptiveCohortOptimization:
    def __init__(
        self, budget, dimension=5, population_size=100, elite_fraction=0.1, mutation_rate=0.1, beta=0.5
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(population_size * elite_fraction)
        self.mutation_rate = mutation_rate
        self.beta = beta  # Mutation shrinkage factor for precise tuning

    def __call__(self, func):
        # Initialize population within the bounds [-5, 5]
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(individual) for individual in population])
        evaluations = self.population_size

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        while evaluations < self.budget:
            new_population = np.empty_like(population)

            # Select elites
            elite_indices = np.argsort(fitness)[: self.elite_count]
            elites = population[elite_indices]

            # Generate new population
            for i in range(self.population_size):
                if np.random.rand() < self.mutation_rate:  # Mutation occurs
                    parent_idx = np.random.choice(self.elite_count)
                    parent = elites[parent_idx]
                    mutation = self.beta * np.random.normal(0, 1, self.dimension)
                    child = parent + mutation
                else:  # Crossover between two elites
                    parents_indices = np.random.choice(elite_indices, 2, replace=False)
                    crossover_point = np.random.randint(self.dimension)
                    child = np.concatenate(
                        (
                            population[parents_indices[0]][:crossover_point],
                            population[parents_indices[1]][crossover_point:],
                        )
                    )

                # Ensure the child is within bounds
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

            # Dynamically adjust mutation rate and beta for fine-tuning
            self.mutation_rate *= 0.98
            self.beta *= 0.95

        return best_fitness, best_individual
