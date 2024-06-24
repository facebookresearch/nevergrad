import numpy as np


class EnhancedAdaptiveDiversifiedEvolutionStrategy:
    def __init__(self, budget, dimension=5, lower_bound=-5.0, upper_bound=5.0, population_size=30):
        self.budget = budget
        self.dimension = dimension
        self.bounds = {"lb": lower_bound, "ub": upper_bound}
        self.population_size = population_size
        self.mutation_scale_base = 0.1  # Base scale for mutation
        self.crossover_probability = 0.7  # Probability of crossover

    def mutate(self, individual):
        """Apply Gaussian mutation with adaptive scaling."""
        scale = self.mutation_scale_base * np.random.rand()
        mutation = np.random.normal(0, scale, self.dimension)
        mutant = individual + mutation
        return np.clip(mutant, self.bounds["lb"], self.bounds["ub"])

    def crossover(self, parent1, parent2):
        """Uniform crossover between two parents."""
        mask = np.random.rand(self.dimension) < 0.5
        offspring = np.where(mask, parent1, parent2)
        return offspring

    def select(self, population, fitness, offspring, offspring_fitness):
        """Tournament selection to decide the next generation."""
        better_mask = offspring_fitness < fitness
        population[better_mask] = offspring[better_mask]
        fitness[better_mask] = offspring_fitness[better_mask]
        return population, fitness

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(
            self.bounds["lb"], self.bounds["ub"], (self.population_size, self.dimension)
        )
        fitness = np.array([func(individual) for individual in population])
        f_opt = np.min(fitness)
        x_opt = population[np.argmin(fitness)]

        # Evolutionary loop
        iterations = self.budget // self.population_size
        for _ in range(iterations):
            offspring = []
            offspring_fitness = []

            # Generate offspring
            for idx in range(self.population_size):
                # Mutation
                mutant = self.mutate(population[idx])

                # Crossover
                if np.random.rand() < self.crossover_probability:
                    partner_idx = np.random.randint(self.population_size)
                    child = self.crossover(mutant, population[partner_idx])
                else:
                    child = mutant

                # Evaluate
                child_fitness = func(child)
                offspring.append(child)
                offspring_fitness.append(child_fitness)

            # Selection
            offspring = np.array(offspring)
            offspring_fitness = np.array(offspring_fitness)
            population, fitness = self.select(population, fitness, offspring, offspring_fitness)

            # Update best found solution
            min_idx = np.argmin(fitness)
            if fitness[min_idx] < f_opt:
                f_opt = fitness[min_idx]
                x_opt = population[min_idx]

        return f_opt, x_opt
