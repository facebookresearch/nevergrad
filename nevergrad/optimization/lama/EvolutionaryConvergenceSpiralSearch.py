import numpy as np


class EvolutionaryConvergenceSpiralSearch:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality as per problem's constraints

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Initialize the centroid and search parameters
        centroid = np.random.uniform(-5.0, 5.0, self.dim)
        radius = 5.0  # Initial radius
        angle_increment = np.pi / 16  # Initial angle increment for exploration

        # Adaptive decay rates for flexibility
        radius_decay = 0.98  # Radius decay rate
        angle_refinement = 0.98  # Angle refinement rate
        evaluations_left = self.budget
        min_radius = 0.0005  # Extremely fine minimum radius for detailed exploration

        # Evolutionary adaptations
        population_size = 15  # Number of points to consider around the centroid
        offspring_size = 10  # Number of new points generated from mutations
        mutation_rate = 0.1  # Mutation rate for generating offspring

        while evaluations_left > 0:
            # Generate a population around the centroid
            population = np.array(
                [centroid + radius * np.random.uniform(-1, 1, self.dim) for _ in range(population_size)]
            )
            population = np.clip(population, -5.0, 5.0)  # Enforce bounds
            fitness = np.array([func(ind) for ind in population])
            evaluations_left -= population_size

            # Select the best individual
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.f_opt:
                self.f_opt = fitness[best_idx]
                self.x_opt = population[best_idx]

            # Generate offspring by mutation
            offspring = np.array(
                [
                    population[best_idx] + mutation_rate * np.random.normal(0, radius, self.dim)
                    for _ in range(offspring_size)
                ]
            )
            offspring = np.clip(offspring, -5.0, 5.0)  # Enforce bounds
            offspring_fitness = np.array([func(ind) for ind in offspring])
            evaluations_left -= offspring_size

            # Update centroid and search parameters
            centroid = offspring[np.argmin(offspring_fitness)]
            radius *= radius_decay
            radius = max(radius, min_radius)
            angle_increment *= angle_refinement

        return self.f_opt, self.x_opt
