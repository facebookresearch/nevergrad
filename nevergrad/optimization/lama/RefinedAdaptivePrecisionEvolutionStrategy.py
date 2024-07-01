import numpy as np


class RefinedAdaptivePrecisionEvolutionStrategy:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=40,
        elite_fraction=0.1,
        mutation_factor=0.8,
        crossover_prob=0.7,
    ):
        self.budget = budget
        self.dimension = dimension
        self.bounds = {"lb": lower_bound, "ub": upper_bound}
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.sigma = 0.5  # Initial standard deviation for Gaussian mutation
        self.learning_rate = 0.2  # Learning rate for sigma adaptation
        self.mutation_factor = mutation_factor  # Factor for differential mutation
        self.crossover_prob = crossover_prob  # Probability of crossover

    def mutate(self, individual, best_individual):
        """Differential evolution mutation"""
        mutation = (
            np.random.normal(0, self.sigma, self.dimension)
            * self.mutation_factor
            * (best_individual - individual)
        )
        return np.clip(individual + mutation, self.bounds["lb"], self.bounds["ub"])

    def crossover(self, parent, donor):
        """Uniform crossover"""
        mask = np.random.rand(self.dimension) < self.crossover_prob
        return np.where(mask, donor, parent)

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
            elites = population[np.argsort(fitness)[:num_elites]]

            # Create offspring using differential mutation and crossover
            offspring = np.zeros_like(population)
            for i in range(self.population_size):
                donor = self.mutate(population[i], best_individual)
                offspring[i] = self.crossover(population[i], donor)

            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations += self.population_size

            # Combine elites and offspring, then select the next generation
            combined_population = np.vstack((elites, offspring[num_elites:]))
            combined_fitness = np.concatenate((fitness[:num_elites], offspring_fitness[num_elites:]))
            indices = np.argsort(combined_fitness)
            population = combined_population[indices]
            fitness = combined_fitness[indices]

            # Update best found solution
            current_best_fitness = np.min(fitness)
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[np.argmin(fitness)]

            # Adapt mutation step size using the success rate of mutations
            successful_mutations = (offspring_fitness < fitness).mean()
            self.sigma *= np.exp(self.learning_rate * (successful_mutations - 0.2) / (1 - 0.2))

            if evaluations >= self.budget:
                break

        return best_fitness, best_individual
