import numpy as np


class CohortDiversityDrivenOptimization:
    def __init__(
        self,
        budget,
        dimension=5,
        population_size=100,
        elite_fraction=0.15,
        mutation_factor=0.08,
        recombination_prob=0.85,
        adaptation_intensity=0.95,
    ):
        self.budget = budget
        self.dimension = dimension
        self.population_size = population_size
        self.elite_count = int(population_size * elite_fraction)
        self.mutation_factor = mutation_factor
        self.recombination_prob = recombination_prob
        self.adaptation_intensity = adaptation_intensity

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
            mean_elite = np.mean(population[elite_indices], axis=0)

            for i in range(self.population_size):
                if np.random.rand() < self.recombination_prob:
                    # Recombination from elite members
                    parents_indices = np.random.choice(elite_indices, 2, replace=False)
                    parent1, parent2 = population[parents_indices[0]], population[parents_indices[1]]
                    alpha = np.random.rand()
                    child = alpha * parent1 + (1 - alpha) * parent2
                else:
                    # Mutation based on distance from elite mean
                    elite_member = population[np.random.choice(elite_indices)]
                    mutation_direction = np.random.randn(self.dimension)
                    child = elite_member + self.mutation_factor * mutation_direction * np.linalg.norm(
                        mean_elite - elite_member
                    )

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

            # Adapt mutation factor to converge slower as nearing budget limit
            self.mutation_factor *= self.adaptation_intensity

        return best_fitness, best_individual
