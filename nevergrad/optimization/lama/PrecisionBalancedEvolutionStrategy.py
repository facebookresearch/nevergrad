import numpy as np


class PrecisionBalancedEvolutionStrategy:
    def __init__(self, budget, dimension=5, lower_bound=-5.0, upper_bound=5.0, population_size=50):
        self.budget = budget
        self.dimension = dimension
        self.bounds = np.array([lower_bound, upper_bound])
        self.population_size = population_size
        self.sigma = 0.5  # Initial standard deviation for mutations
        self.learning_rate = 0.1  # Learning rate for self-adaptation of sigma

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def mutate(self, individual):
        # Mutation with adaptive sigma
        return np.clip(
            individual + np.random.normal(0, self.sigma, self.dimension), self.bounds[0], self.bounds[1]
        )

    def recombine(self, parents):
        # Intermediate recombination
        return np.mean(parents, axis=0)

    def select(self, population, fitness):
        # Select the best individual
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

    def adapt_sigma(self, success_rate):
        # Adapt sigma based on success rate
        if success_rate > 0.2:
            self.sigma /= self.learning_rate
        elif success_rate < 0.2:
            self.sigma *= self.learning_rate

    def optimize(self, func):
        population = self.initialize_population()
        fitness = np.array([func(ind) for ind in population])
        best_individual, best_fitness = self.select(population, fitness)

        evaluations = self.population_size
        successful_mutations = 0

        while evaluations < self.budget:
            for i in range(self.population_size):
                mutant = self.mutate(population[i])
                mutant_fitness = func(mutant)

                if mutant_fitness < fitness[i]:
                    population[i] = mutant
                    fitness[i] = mutant_fitness
                    successful_mutations += 1

                if mutant_fitness < best_fitness:
                    best_individual = mutant
                    best_fitness = mutant_fitness

                evaluations += 1
                if evaluations >= self.budget:
                    break

            # Adapt sigma based on the success rate of mutations
            success_rate = successful_mutations / self.population_size
            self.adapt_sigma(success_rate)
            successful_mutations = 0  # Reset for next generation

        return best_fitness, best_individual

    def __call__(self, func):
        return self.optimize(func)
