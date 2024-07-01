import numpy as np


class HybridAdaptiveCrossoverElitistStrategyV10:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=300,
        elite_fraction=0.1,
        mutation_intensity=0.02,
        crossover_rate=0.88,
        adaptive_intensity=0.8,
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(population_size * elite_fraction)
        self.mutation_intensity = mutation_intensity
        self.crossover_rate = crossover_rate
        self.adaptive_intensity = adaptive_intensity

    def __call__(self, func):
        # Initialize the population
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(individual) for individual in population])
        evaluations = self.population_size

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        while evaluations < self.budget:
            # Select elites
            elite_indices = np.argsort(fitness)[: self.elite_count]
            elites = population[elite_indices]

            # Generate new population
            new_population = np.empty_like(population)
            for i in range(self.population_size):
                if np.random.random() < self.crossover_rate:
                    # Perform hybrid crossover
                    parent1, parent2 = elites[np.random.choice(len(elites), 2, replace=False)]
                    child = self.hybrid_recombine(parent1, parent2, evaluations)
                else:
                    # Mutation of an elite
                    parent = elites[np.random.choice(len(elites))]
                    child = self.adaptive_mutate(parent, evaluations)

                new_population[i] = np.clip(child, -5.0, 5.0)
                new_fitness = func(new_population[i])
                evaluations += 1

                # Update the best solution found
                if new_fitness < best_fitness:
                    best_fitness = new_fitness
                    best_individual = new_population[i]

                if evaluations >= self.budget:
                    break

            # Replace old population
            population = new_population
            fitness = np.array([func(individual) for individual in population])

        return best_fitness, best_individual

    def adaptive_mutate(self, individual, evaluations):
        # Adaptive mutation intensity based on normalized evaluations
        normalized_time = evaluations / self.budget
        intensity = self.mutation_intensity * np.exp(-normalized_time * 10)
        return individual + np.random.normal(0, intensity, self.dimension)

    def hybrid_recombine(self, parent1, parent2, evaluations):
        # Blend between parents with adaptive depth based on evaluations
        normalized_time = evaluations / self.budget
        alpha = (
            np.random.uniform(0.3, 0.7) * (1 - normalized_time) + normalized_time * self.adaptive_intensity
        )
        return alpha * parent1 + (1 - alpha) * parent2
