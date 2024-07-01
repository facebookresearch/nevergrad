import numpy as np


class EnhancedPrecisionTunedCrossoverElitistStrategyV14:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=600,
        elite_fraction=0.3,
        mutation_intensity=0.025,
        crossover_rate=0.85,
        adaptivity_coefficient=0.9,
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(population_size * elite_fraction)
        self.mutation_intensity = mutation_intensity
        self.crossover_rate = crossover_rate
        self.adaptivity_coefficient = adaptivity_coefficient

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
                parent1 = elites[np.random.choice(len(elites))]
                if np.random.random() < self.crossover_rate:
                    # Perform crossover
                    parent2 = elites[np.random.choice(len(elites))]
                    child = self.adaptive_crossover(parent1, parent2, evaluations)
                else:
                    # Mutation of an elite
                    child = self.adaptive_mutate(parent1, evaluations)

                new_population[i] = np.clip(child, -5.0, 5.0)

            # Evaluate new population
            for i in range(self.population_size):
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
        # Adaptive mutation intensity based on stage of optimization process
        normalized_time = evaluations / self.budget
        intensity = self.mutation_intensity * (1 - np.exp(-self.adaptivity_coefficient * normalized_time))
        return individual + np.random.normal(0, intensity, self.dimension)

    def adaptive_crossover(self, parent1, parent2, evaluations):
        # Adaptive weighted crossover using a dynamic strategy based on evaluations
        normalized_time = evaluations / self.budget
        weight = 0.5 + 0.5 * np.sin(np.pi * normalized_time)
        return weight * parent1 + (1 - weight) * parent2
