import numpy as np


class FineTunedCohortDiversityOptimizer:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=100,
        elite_fraction=0.1,
        mutation_intensity=0.05,
        recombination_prob=0.8,
        adaptation_rate=0.95,
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(population_size * elite_fraction)
        self.mutation_intensity = mutation_intensity
        self.recombination_prob = recombination_prob
        self.adaptation_rate = adaptation_rate

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
            elite_indices = np.argsort(fitness)[: self.elite_count]

            for i in range(self.population_size):
                # Selecting parents using elite indices
                if np.random.rand() < self.recombination_prob:
                    parents_indices = np.random.choice(elite_indices, 2, replace=False)
                    parent1, parent2 = population[parents_indices[0]], population[parents_indices[1]]
                    mask = np.random.rand(self.dimension) < 0.5
                    child = np.where(mask, parent1, parent2)
                else:
                    child = population[
                        np.random.choice(elite_indices)
                    ].copy()  # Inherit directly from a single elite

                # Mutation: perturb the offspring
                mutation = np.random.normal(scale=self.mutation_intensity, size=self.dimension)
                child = np.clip(child + mutation, -5.0, 5.0)

                child_fitness = func(child)
                evaluations += 1

                if child_fitness < best_fitness:
                    best_fitness = child_fitness
                    best_individual = child

                new_population[i] = child

                if evaluations >= self.budget:
                    break

            population = new_population
            fitness = np.array([func(individual) for individual in population])

            # Adaptively update mutation intensity
            self.mutation_intensity *= self.adaptation_rate

        return best_fitness, best_individual
