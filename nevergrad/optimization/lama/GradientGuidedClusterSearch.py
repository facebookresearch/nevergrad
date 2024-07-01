import numpy as np


class GradientGuidedClusterSearch:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5  # as per problem statement
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Initial population
        population_size = 15
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dimension))
        fitness = np.array([func(individual) for individual in population])

        # Main algorithm loop
        iteration = 0
        while iteration < self.budget:
            best_idx = np.argmin(fitness)
            best_individual = population[best_idx]

            if fitness[best_idx] < self.f_opt:
                self.f_opt = fitness[best_idx]
                self.x_opt = best_individual

            # Approximate gradient calculation using finite differences
            gradients = []
            for i in range(population_size):
                grad = np.zeros(self.dimension)
                for d in range(self.dimension):
                    perturb = np.zeros(self.dimension)
                    perturb[d] = 0.01  # Perturbation value
                    perturbed_individual = population[i] + perturb
                    perturbed_individual = np.clip(perturbed_individual, self.lower_bound, self.upper_bound)
                    perturbed_fitness = func(perturbed_individual)
                    grad[d] = (perturbed_fitness - fitness[i]) / 0.01
                gradients.append(grad)

            # Use gradients to adjust positions
            new_population = []
            for i in range(population_size):
                step_size = 0.1 * (self.upper_bound - self.lower_bound)
                new_individual = population[i] - step_size * gradients[i]
                new_individual = np.clip(new_individual, self.lower_bound, self.upper_bound)
                new_fitness = func(new_individual)

                if new_fitness < fitness[i]:
                    new_population.append(new_individual)
                    fitness[i] = new_fitness
                else:
                    # Exploration with random mutation
                    random_individual = population[i] + np.random.normal(0, 1, self.dimension)
                    random_individual = np.clip(random_individual, self.lower_bound, self.upper_bound)
                    random_fitness = func(random_individual)
                    if random_fitness < fitness[i]:
                        new_population.append(random_individual)
                        fitness[i] = random_fitness
                    else:
                        new_population.append(population[i])

            population = np.array(new_population)
            iteration += population_size

        return self.f_opt, self.x_opt
