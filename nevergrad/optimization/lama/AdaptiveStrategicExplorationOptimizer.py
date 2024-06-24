import numpy as np


class AdaptiveStrategicExplorationOptimizer:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=50,
        elite_fraction=0.1,
        mutation_intensity=0.5,
        crossover_rate=0.7,
    ):
        self.budget = budget
        self.dimension = dimension
        self.bounds = {"lb": lower_bound, "ub": upper_bound}
        self.population_size = population_size
        self.elite_fraction = elite_fraction
        self.mutation_intensity = mutation_intensity  # Initial intensity of mutation
        self.crossover_rate = crossover_rate  # Crossover probability
        self.sigma = 0.1  # Initial standard deviation for normal distribution in mutation

    def mutate(self, individual):
        """Adaptive mutation based on a decreasing strategy"""
        mutation_scale = self.mutation_intensity * (self.budget - self.evaluations) / self.budget
        mutation = individual + mutation_scale * np.random.randn(self.dimension)
        return np.clip(mutation, self.bounds["lb"], self.bounds["ub"])

    def crossover(self, parent, donor):
        """Blended crossover which can adjust the influence of each parent"""
        alpha = np.random.uniform(-0.1, 1.1, size=self.dimension)
        offspring = alpha * parent + (1 - alpha) * donor
        return np.clip(offspring, self.bounds["lb"], self.bounds["ub"])

    def __call__(self, func):
        # Initialize the population
        population = np.random.uniform(
            self.bounds["lb"], self.bounds["ub"], (self.population_size, self.dimension)
        )
        fitness = np.array([func(ind) for ind in population])
        best_fitness = np.min(fitness)
        best_individual = population[np.argmin(fitness)]
        self.evaluations = self.population_size

        while self.evaluations < self.budget:
            # Elite selection
            num_elites = int(self.population_size * self.elite_fraction)
            elites_indices = np.argsort(fitness)[:num_elites]
            elites = population[elites_indices]

            # Generate offspring using mutation and crossover
            new_population = np.empty_like(population)
            for i in range(self.population_size):
                if i < num_elites:
                    # Preserve elites without changes
                    new_population[i] = elites[i]
                else:
                    # select random elite for crossover
                    elite = elites[np.random.randint(num_elites)]
                    mutated = self.mutate(population[np.random.randint(self.population_size)])
                    new_population[i] = self.crossover(elite, mutated)

            # Evaluate new population
            new_fitness = np.array([func(ind) for ind in new_population])
            self.evaluations += self.population_size

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

        return best_fitness, best_individual
