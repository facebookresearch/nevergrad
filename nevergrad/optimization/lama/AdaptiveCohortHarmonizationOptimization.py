import numpy as np


class AdaptiveCohortHarmonizationOptimization:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=100,
        elite_fraction=0.1,
        mutation_intensity=0.05,
        crossover_rate=0.7,
        adaptive_intensity=0.95,
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(population_size * elite_fraction)
        self.mutation_intensity = mutation_intensity
        self.crossover_rate = crossover_rate
        self.adaptive_intensity = adaptive_intensity

    def __call__(self, func):
        # Initialize population within the bounds [-5, 5]
        population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dimension))
        fitness = np.array([func(x) for x in population])
        evaluations = self.population_size

        best_idx = np.argmin(fitness)
        best_individual = population[best_idx]
        best_fitness = fitness[best_idx]

        while evaluations < self.budget:
            new_population = np.empty_like(population)
            elite_indices = np.argsort(fitness)[: self.elite_count]
            elite_population = population[elite_indices]
            global_best = population[elite_indices[0]]

            for i in range(self.population_size):
                if np.random.rand() < self.crossover_rate:
                    # Crossover from elite and a random member
                    random_member = population[np.random.randint(0, self.population_size)]
                    crossover_point = np.random.randint(1, self.dimension)
                    child = np.concatenate((global_best[:crossover_point], random_member[crossover_point:]))
                else:
                    # Mutation based on current member and global best
                    current_member = population[np.random.randint(0, self.population_size)]
                    mutation_vector = self.mutation_intensity * (global_best - current_member)
                    child = current_member + mutation_vector

                # Ensure child stays within bounds
                child = np.clip(child, -5.0, 5.0)
                child_fitness = func(child)
                evaluations += 1

                if child_fitness < best_fitness:
                    best_fitness = child_fitness
                    best_individual = child

                new_population[i] = child

                if evaluations >= self.budget:
                    break

            population = new_population
            fitness = np.array([func(x) for x in population])

            # Adaptively adjust mutation intensity and crossover rate
            self.mutation_intensity *= self.adaptive_intensity
            self.crossover_rate = min(self.crossover_rate + 0.01, 1.0)

        return best_fitness, best_individual
