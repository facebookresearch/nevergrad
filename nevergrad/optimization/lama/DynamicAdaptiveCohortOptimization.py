import numpy as np


class DynamicAdaptiveCohortOptimization:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=100,
        elite_fraction=0.2,
        mutation_base=0.1,
        recombination_prob=0.9,
        adaptation_factor=0.98,
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(population_size * elite_fraction)
        self.mutation_base = mutation_base
        self.recombination_prob = recombination_prob
        self.adaptation_factor = adaptation_factor

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
                if np.random.rand() < self.recombination_prob:
                    # Select parents from elite group
                    parents_indices = np.random.choice(elite_indices, 2, replace=False)
                    parent1, parent2 = population[parents_indices[0]], population[parents_indices[1]]
                    mask = np.random.rand(self.dimension) < 0.5
                    child = np.where(mask, parent1, parent2)
                else:
                    # Inherit directly from an elite member
                    child = population[np.random.choice(elite_indices)].copy()

                # Dynamic mutation based on how far the process has gone
                mutation_scale = self.mutation_base * (1 - evaluations / self.budget) ** 2
                mutation = np.random.normal(scale=mutation_scale, size=self.dimension)
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

            # Adapt mutation base
            self.mutation_base *= self.adaptation_factor

        return best_fitness, best_individual
