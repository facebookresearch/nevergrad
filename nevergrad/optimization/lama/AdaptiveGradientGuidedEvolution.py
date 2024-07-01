import numpy as np


class AdaptiveGradientGuidedEvolution:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=50,
        mutation_intensity=0.1,
        gradient_sampling=10,
    ):
        self.budget = budget
        self.dimension = dimension
        self.bounds = np.array([lower_bound, upper_bound])
        self.population_size = population_size
        self.mutation_intensity = mutation_intensity
        self.gradient_sampling = gradient_sampling  # Number of points to estimate gradient
        self.sigma = 0.2  # Standard deviation for mutations

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def mutate(self, individual):
        # Mutation operation
        return np.clip(
            individual + np.random.normal(0, self.sigma, self.dimension), self.bounds[0], self.bounds[1]
        )

    def approximate_gradient(self, individual, func):
        # Approximate gradient by sampling around the individual
        gradients = []
        initial_fitness = func(individual)
        for _ in range(self.gradient_sampling):
            perturbation = np.random.normal(0, self.sigma, self.dimension)
            neighbor = np.clip(individual + perturbation, self.bounds[0], self.bounds[1])
            neighbor_fitness = func(neighbor)
            gradient = (
                (neighbor_fitness - initial_fitness) / (np.linalg.norm(perturbation) + 1e-6) * perturbation
            )
            gradients.append(gradient)
        return np.mean(gradients, axis=0)

    def optimize(self, func):
        population = self.initialize_population()
        fitness = np.array([func(ind) for ind in population])
        best_individual, best_fitness = population[np.argmin(fitness)], np.min(fitness)

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                gradient = self.approximate_gradient(population[i], func)
                individual = population[i] - self.mutation_intensity * gradient  # Gradient descent step
                individual = self.mutate(individual)  # Mutation step
                individual_fitness = func(individual)
                evaluations += 1

                if individual_fitness < fitness[i]:
                    population[i] = individual
                    fitness[i] = individual_fitness

                if individual_fitness < best_fitness:
                    best_individual = individual
                    best_fitness = individual_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_individual

    def __call__(self, func):
        return self.optimize(func)
