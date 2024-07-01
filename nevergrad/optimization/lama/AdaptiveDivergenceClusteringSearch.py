import numpy as np


class AdaptiveDivergenceClusteringSearch:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5  # given as per problem statement
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Initialize population
        population_size = 10
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dimension))

        # Evaluate initial population
        fitness = np.array([func(individual) for individual in population])

        # Main optimization loop
        iteration = 0
        while iteration < self.budget:
            # Select the best solution for breeding
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]

            # Update optimal solution
            if fitness[best_idx] < self.f_opt:
                self.f_opt = fitness[best_idx]
                self.x_opt = best_individual

            # Generate new solutions based on adaptive divergence and clustering
            new_population = []
            cluster_center = np.mean(population, axis=0)
            for i in range(population_size):
                if np.random.rand() < 0.5:
                    # Diverge away from the cluster center
                    new_individual = population[i] + np.random.normal(0, 1, self.dimension) * (
                        population[i] - cluster_center
                    )
                else:
                    # Converge towards the best solution
                    new_individual = population[i] + np.random.normal(0, 1, self.dimension) * (
                        best_individual - population[i]
                    )

                # Ensure new individual is within bounds
                new_individual = np.clip(new_individual, self.lower_bound, self.upper_bound)
                new_population.append(new_individual)

            # Evaluate new population
            new_fitness = np.array([func(individual) for individual in new_population])

            # Replace old population with new if better
            for i in range(population_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]

            iteration += population_size

        return self.f_opt, self.x_opt
