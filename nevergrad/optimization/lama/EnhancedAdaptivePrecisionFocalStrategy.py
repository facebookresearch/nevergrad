import numpy as np


class EnhancedAdaptivePrecisionFocalStrategy:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=100,
        focal_ratio=0.1,
        elite_ratio=0.05,
    ):
        self.budget = budget
        self.dimension = dimension
        self.bounds = np.array([lower_bound, upper_bound])
        self.population_size = population_size
        self.focal_population_size = int(population_size * focal_ratio)
        self.elite_population_size = int(population_size * elite_ratio)
        self.sigma = 0.3  # Initial standard deviation for mutations
        self.learning_rate = 0.1  # Learning rate for self-adaptation of sigma

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def mutate(self, individual):
        # Mutation with adaptive sigma
        return np.clip(
            individual + np.random.normal(0, self.sigma, self.dimension), self.bounds[0], self.bounds[1]
        )

    def select_focal_group(self, population, fitness):
        # Select a smaller focal group based on the best fitness values
        sorted_indices = np.argsort(fitness)
        return population[sorted_indices[: self.focal_population_size]]

    def select_elite_group(self, population, fitness):
        # Select the elite group for intense exploitation
        sorted_indices = np.argsort(fitness)
        return population[sorted_indices[: self.elite_population_size]]

    def recombine(self, focal_group):
        # Global intermediate recombination from a focal group
        return np.mean(focal_group, axis=0)

    def adapt_sigma(self, success_rate):
        # Dynamically adjust sigma based on observed mutation success
        if success_rate > 0.2:
            self.sigma /= self.learning_rate
        elif success_rate < 0.2:
            self.sigma *= self.learning_rate

    def optimize(self, func):
        population = self.initialize_population()
        fitness = np.array([func(ind) for ind in population])
        best_individual, best_fitness = population[np.argmin(fitness)], np.min(fitness)

        evaluations = self.population_size
        successful_mutations = 0

        while evaluations < self.budget:
            focal_group = self.select_focal_group(population, fitness)
            elite_group = self.select_elite_group(population, fitness)
            recombined_individual = self.recombine(focal_group)

            for i in range(self.population_size):
                if i < self.elite_population_size:
                    mutant = self.mutate(elite_group[i % self.elite_population_size])
                else:
                    mutant = self.mutate(recombined_individual)

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

            # Adjust mutation strategy based on success
            success_rate = successful_mutations / self.population_size
            self.adapt_sigma(success_rate)
            successful_mutations = 0  # Reset for next generation

        return best_fitness, best_individual

    def __call__(self, func):
        return self.optimize(func)
