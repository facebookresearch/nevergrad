import numpy as np


class ALES:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        population_size = 50
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dimension))
        fitness = np.array([func(x) for x in population])
        num_evals = population_size

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()

        # Evolution parameters
        learning_rate = 0.1
        global_learning_rate = 0.2
        mutation_strength = 0.5
        mutation_decay = 0.99
        elite_fraction = 0.1
        elite_size = max(int(population_size * elite_fraction), 1)

        while num_evals < self.budget:
            elite_indices = np.argsort(fitness)[:elite_size]
            global_mean = np.mean(population[elite_indices], axis=0)

            for i in range(population_size):
                if num_evals >= self.budget:
                    break

                if i in elite_indices:
                    # Elites undergo less mutation
                    step = mutation_strength * np.random.randn(self.dimension) * 0.5
                else:
                    step = mutation_strength * np.random.randn(self.dimension)

                individual = population[i] + step
                individual = np.clip(individual, self.lower_bound, self.upper_bound)

                # Perform a global pull move towards the mean of elites
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

            # Decay the mutation strength
            mutation_strength *= mutation_decay

        return best_fitness, best_individual
