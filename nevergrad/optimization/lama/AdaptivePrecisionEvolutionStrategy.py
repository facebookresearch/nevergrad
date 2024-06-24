import numpy as np


class AdaptivePrecisionEvolutionStrategy:
    def __init__(
        self, budget, dimension=5, lower_bound=-5.0, upper_bound=5.0, population_size=40, elite_fraction=0.1
    ):
        self.budget = budget
        self.dimension = dimension
        self.bounds = {"lb": lower_bound, "ub": upper_bound}
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.sigma = 0.5  # Initial standard deviation for Gaussian mutation
        self.learning_rate = 0.2  # Learning rate for sigma adaptation

    def mutate(self, individual):
        """Gaussian mutation"""
        mutation = np.random.normal(0, self.sigma, self.dimension)
        return np.clip(individual + mutation, self.bounds["lb"], self.bounds["ub"])

    def select_elites(self, population, fitness, num_elites):
        """Select elite individuals"""
        elite_indices = np.argsort(fitness)[:num_elites]
        return population[elite_indices]

    def __call__(self, func):
        # Initialize population uniformly within the bounds
        population = np.random.uniform(
            self.bounds["lb"], self.bounds["ub"], (self.population_size, self.dimension)
        )
        fitness = np.array([func(ind) for ind in population])
        best_fitness = np.min(fitness)
        best_individual = population[np.argmin(fitness)]
        evaluations = self.population_size

        while evaluations < self.budget:
            num_elites = int(self.population_size * self.elite_fraction)
            elites = self.select_elites(population, fitness, num_elites)

            # Create the offspring by mutation
            offspring = np.array([self.mutate(ind) for ind in population])
            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += self.population_size

            # Combine and select the next generation
            combined_population = np.vstack((elites, offspring[num_elites:]))
            combined_fitness = np.concatenate((fitness[:num_elites], offspring_fitness[num_elites:]))

            # Environment selection
            selection_indices = np.argsort(combined_fitness)[: self.population_size]
            population = combined_population[selection_indices]
            fitness = combined_fitness[selection_indices]

            # Update best solution found
            current_best_fitness = np.min(fitness)
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[np.argmin(fitness)]

            # Adapt mutation step size
            successful_mutations = (offspring_fitness < fitness).mean()
            self.sigma *= np.exp(self.learning_rate * (successful_mutations - 0.2) / (1 - 0.2))

            if evaluations >= self.budget:
                break

        return best_fitness, best_individual
