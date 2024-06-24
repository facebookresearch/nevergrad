import numpy as np


class ConcentricGradientEnhancedEvolver:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=100,
        perturbation_scale=0.1,
        learning_rate=0.01,
    ):
        self.budget = budget
        self.dimension = dimension
        self.bounds = np.array([lower_bound, upper_bound])
        self.population_size = population_size
        self.perturbation_scale = perturbation_scale  # Controls perturbation for gradient estimation
        self.learning_rate = learning_rate  # Controls the step size in the gradient update

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def mutate(self, individual):
        # Mutation operation using Gaussian noise to maintain diversity
        mutation = np.random.normal(0, self.perturbation_scale, self.dimension)
        return np.clip(individual + mutation, self.bounds[0], self.bounds[1])

    def approximate_gradient(self, individual, func):
        # Improved gradient estimation with central difference
        gradients = np.zeros(self.dimension)
        for idx in range(self.dimension):
            perturbation = np.zeros(self.dimension)
            perturbation[idx] = self.perturbation_scale
            forward = np.clip(individual + perturbation, self.bounds[0], self.bounds[1])
            backward = np.clip(individual - perturbation, self.bounds[0], self.bounds[1])
            gradients[idx] = (func(forward) - func(backward)) / (2 * self.perturbation_scale)
        return gradients

    def optimize(self, func):
        population = self.initialize_population()
        fitness = np.array([func(ind) for ind in population])
        best_individual, best_fitness = population[np.argmin(fitness)], np.min(fitness)

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                gradient = self.approximate_gradient(population[i], func)
                new_individual = (
                    population[i] - self.learning_rate * gradient
                )  # Gradient descent step with controlled learning rate
                new_individual = self.mutate(new_individual)  # Apply mutation for diversity
                new_fitness = func(new_individual)
                evaluations += 1

                if new_fitness < fitness[i]:
                    population[i] = new_individual
                    fitness[i] = new_fitness

                if new_fitness < best_fitness:
                    best_individual = new_individual
                    best_fitness = new_fitness

                if evaluations >= self.budget:
                    break

        return best_fitness, best_individual

    def __call__(self, func):
        return self.optimize(func)
