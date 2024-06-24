import numpy as np


class RefinedAdaptivePrecisionCohortOptimizationV4:
    def __init__(self, budget, dimension=5, population_size=200, elite_fraction=0.2, mutation_intensity=0.8):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(population_size * elite_fraction)
        self.mutation_intensity = mutation_intensity  # Base intensity for mutation

    def __call__(self, func):
        # Initialize the population uniformly within the search space [-5, 5]
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(individual) for individual in population])
        evaluations = self.population_size

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        while evaluations < self.budget:
            new_population = np.empty_like(population)

            # Select the elite individuals
            elite_indices = np.argsort(fitness)[: self.elite_count]
            elites = population[elite_indices]

            # Generate new population members
            for i in range(self.population_size):
                if np.random.rand() < self.adaptive_mutation_rate(evaluations):
                    # Mutation: pick a random elite, apply Gaussian noise
                    parent_idx = np.random.choice(elite_indices)
                    mutation = np.random.normal(0, self.adaptive_mutation_scale(evaluations), self.dimension)
                    child = np.clip(population[parent_idx] + mutation, -5.0, 5.0)
                else:
                    # Crossover: pick two different elites, combine their features
                    parents = np.random.choice(elite_indices, 2, replace=False)
                    crossover_point = np.random.randint(1, self.dimension)
                    child = np.concatenate(
                        (population[parents[0], :crossover_point], population[parents[1], crossover_point:])
                    )

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

        return best_fitness, best_individual

    def adaptive_mutation_rate(self, evaluations):
        # Gradually decrease mutation rate to allow more exploration initially
        return max(0.1, 1 - (evaluations / self.budget))

    def adaptive_mutation_scale(self, evaluations):
        # Decay mutation scale logarithmically to fine-tune search in later stages
        return self.mutation_intensity * (1 / (1 + np.log(1 + evaluations)))
