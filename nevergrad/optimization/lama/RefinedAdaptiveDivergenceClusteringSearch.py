import numpy as np


class RefinedAdaptiveDivergenceClusteringSearch:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5  # given as per problem statement
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Initialize population
        population_size = 20
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dimension))
        fitness = np.array([func(individual) for individual in population])

        # Initialize tracking for adaptive mechanism
        last_improvement = 0
        no_improvement_stretch = 0

        iteration = 0
        while iteration < self.budget:
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]

            if fitness[best_idx] < self.f_opt:
                self.f_opt = fitness[best_idx]
                self.x_opt = best_individual
                last_improvement = iteration

            no_improvement_stretch = iteration - last_improvement

            # Calculate adaptive exploration rate
            exploration_rate = 0.5 * np.exp(-no_improvement_stretch / 50)

            new_population = []
            for i in range(population_size):
                if np.random.rand() < exploration_rate:
                    # Divergence logic
                    random_direction = np.random.normal(0, 1, self.dimension)
                    new_individual = best_individual + random_direction * np.random.rand()
                else:
                    # Convergence logic
                    convergence_factor = np.random.uniform(0.1, 0.5)
                    new_individual = best_individual + convergence_factor * (population[i] - best_individual)

                # Local search mechanism
                local_search_step = 0.1 * (self.upper_bound - self.lower_bound)
                local_individual = new_individual + np.random.uniform(
                    -local_search_step, local_search_step, self.dimension
                )
                local_individual = np.clip(local_individual, self.lower_bound, self.upper_bound)

                # Selection between new individual and local search result based on fitness
                new_individual_fitness = func(new_individual)
                local_individual_fitness = func(local_individual)

                if local_individual_fitness < new_individual_fitness:
                    new_population.append(local_individual)
                    new_fitness = local_individual_fitness
                else:
                    new_population.append(new_individual)
                    new_fitness = new_individual_fitness

                # Update best found solution
                if new_fitness < self.f_opt:
                    self.f_opt = new_fitness
                    self.x_opt = new_population[-1]

            population = np.array(new_population)
            fitness = np.array([func(individual) for individual in population])

            iteration += population_size

        return self.f_opt, self.x_opt
