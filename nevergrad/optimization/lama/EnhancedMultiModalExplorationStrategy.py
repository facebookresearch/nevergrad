import numpy as np


class EnhancedMultiModalExplorationStrategy:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=50,
        elite_fraction=0.2,
        mutation_intensity=1.0,
        crossover_rate=0.8,
    ):
        self.budget = budget
        self.dimension = dimension
        self.bounds = {"lb": lower_bound, "ub": upper_bound}
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.mutation_intensity = mutation_intensity  # Intensity of mutation
        self.crossover_rate = crossover_rate  # Crossover probability
        self.sigma = 0.3  # Standard deviation of Gaussian noise for mutation

    def mutate(self, individual):
        """Hybrid mutation incorporating both global and local search tendencies"""
        global_mutation = individual + self.mutation_intensity * np.random.randn(self.dimension)
        local_mutation = individual + np.random.normal(0, self.sigma, self.dimension)
        # Choose between global or local mutation based on a random choice
        if np.random.rand() < 0.5:
            return np.clip(global_mutation, self.bounds["lb"], self.bounds["ub"])
        else:
            return np.clip(local_mutation, self.bounds["lb"], self.bounds["ub"])

    def crossover(self, parent, donor):
        """Uniform crossover with adjustable rate"""
        mask = np.random.rand(self.dimension) < self.crossover_rate
        return np.where(mask, donor, parent)

    def __call__(self, func):
        # Initialize the population
        population = np.random.uniform(
            self.bounds["lb"], self.bounds["ub"], (self.population_size, self.dimension)
        )
        fitness = np.array([func(ind) for ind in population])
        best_fitness = np.min(fitness)
        best_individual = population[np.argmin(fitness)]
        evaluations = self.population_size

        while evaluations < self.budget:
            # Elite selection
            num_elites = int(self.population_size * self.elite_fraction)
            elites_indices = np.argsort(fitness)[:num_elites]
            elites = population[elites_indices]

            # Generate offspring using mutation and crossover
            new_population = np.empty_like(population)
            for i in range(self.population_size):
                if i < num_elites:
                    new_population[i] = elites[i]
                else:
                    # select random elite for crossover
                    elite = elites[np.random.randint(num_elites)]
                    mutated = self.mutate(population[i])
                    new_population[i] = self.crossover(elite, mutated)

            # Evaluate new population
            new_fitness = np.array([func(ind) for ind in new_population])
            evaluations += self.population_size

            # Select the best solutions to form the new population
            combined_population = np.vstack((population, new_population))
            combined_fitness = np.concatenate((fitness, new_fitness))
            indices = np.argsort(combined_fitness)[: self.population_size]
            population = combined_population[indices]
            fitness = combined_fitness[indices]

            # Update the best solution found
            current_best_idx = np.argmin(fitness)
            current_best_fitness = fitness[current_best_idx]
            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_individual = population[current_best_idx]

            # Adapt mutation parameters
            successful_ratio = np.mean(new_fitness < fitness)
            self.sigma *= np.exp(0.1 * (successful_ratio - 0.2))

            if evaluations >= self.budget:
                break

        return best_fitness, best_individual
