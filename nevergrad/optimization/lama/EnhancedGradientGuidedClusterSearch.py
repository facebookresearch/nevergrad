import numpy as np


class EnhancedGradientGuidedClusterSearch:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5  # as per problem statement
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Initial population
        population_size = 20
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dimension))
        fitness = np.array([func(individual) for individual in population])

        # Parameters for adaptive mechanisms
        mutation_rate = 0.1
        mutation_scale = 1.0

        iteration = 0
        while iteration < self.budget:
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]

            if fitness[best_idx] < self.f_opt:
                self.f_opt = fitness[best_idx]
                self.x_opt = best_individual

            # Estimating gradients with central difference method for more accuracy
            gradients = []
            for i in range(population_size):
                grad = np.zeros(self.dimension)
                for d in range(self.dimension):
                    perturb = np.zeros(self.dimension)
                    epsilon = 0.01
                    perturb[d] = epsilon
                    forward = np.clip(population[i] + perturb, self.lower_bound, self.upper_bound)
                    backward = np.clip(population[i] - perturb, self.lower_bound, self.upper_bound)
                    grad[d] = (func(forward) - func(backward)) / (2 * epsilon)
                gradients.append(grad)

            # Adaptive gradient step and mutation
            new_population = []
            for i in range(population_size):
                learning_rate = mutation_scale / (1 + iteration / self.budget * 10)  # Decreases over time
                new_individual = population[i] - learning_rate * gradients[i]
                new_individual = np.clip(new_individual, self.lower_bound, self.upper_bound)
                new_fitness = func(new_individual)

                # Mutation with adaptive rate
                if np.random.rand() < mutation_rate:
                    new_individual += np.random.normal(0, mutation_scale, self.dimension)
                    new_individual = np.clip(new_individual, self.lower_bound, self.upper_bound)
                    new_fitness = func(new_individual)

                if new_fitness < fitness[i]:
                    new_population.append(new_individual)
                    fitness[i] = new_fitness
                else:
                    new_population.append(population[i])

            population = np.array(new_population)
            iteration += population_size

            # Adjust mutation parameters based on progress
            if iteration % 100 == 0:
                mutation_rate *= 0.95  # Gradual decrease
                mutation_scale *= 0.99  # Reduce mutation impact over time

        return self.f_opt, self.x_opt
