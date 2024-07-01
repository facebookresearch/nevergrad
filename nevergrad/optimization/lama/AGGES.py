import numpy as np


class AGGES:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population_size = 150  # Further increased population size for broader exploration
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dimension))
        fitness = np.array([func(x) for x in population])
        num_evals = population_size

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()

        # Evolution parameters
        learning_rate = 0.01  # Slight decrease in individual learning rate for finer adjustments
        global_learning_rate = (
            0.3  # Slightly increased to pull population towards better regions more strongly
        )
        mutation_strength = 1.0  # Increased mutation strength for initial broader exploration
        mutation_decay = 0.98  # Slightly slower decay rate
        elite_fraction = 0.15  # Adjust elite fraction for a balance between exploration and exploitation
        elite_size = int(population_size * elite_fraction)

        while num_evals < self.budget:
            elite_indices = np.argsort(fitness)[:elite_size]
            global_mean = np.mean(population[elite_indices], axis=0)

            for i in range(population_size):
                if num_evals >= self.budget:
                    break

                step = mutation_strength * (
                    np.random.randn(self.dimension) + learning_rate * (global_mean - population[i])
                )
                individual = population[i] + step
                individual = np.clip(individual, self.lower_bound, self.upper_bound)

                # Stronger pull towards global mean to accelerate convergence
                individual = individual + global_learning_rate * (global_mean - individual)
                individual_fitness = func(individual)
                num_evals += 1

                # Selection process
                if individual_fitness < fitness[i]:
                    population[i] = individual
                    fitness[i] = individual_fitness
                    if individual_fitness < best_fitness:
                        best_fitness = individual_fitness
                        best_individual = individual.copy()

            # Update mutation strength adaptively based on elite performance improvement
            mutation_strength *= mutation_decay

        return best_fitness, best_individual
