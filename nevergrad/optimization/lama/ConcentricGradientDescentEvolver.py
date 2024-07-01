import numpy as np


class ConcentricGradientDescentEvolver:
    def __init__(
        self,
        budget,
        dimension=5,
        lower_bound=-5.0,
        upper_bound=5.0,
        population_size=100,
        perturbation_scale=0.1,
    ):
        self.budget = budget
        self.dimension = dimension
        self.bounds = np.array([lower_bound, upper_bound])
        self.population_size = population_size
        self.perturbation_scale = perturbation_scale  # Controls perturbation for gradient estimation

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dimension))

    def mutate(self, individual):
        # Mutation operation using Gaussian noise
        mutation = np.random.normal(0, self.perturbation_scale, self.dimension)
        return np.clip(individual + mutation, self.bounds[0], self.bounds[1])

    def approximate_gradient(self, individual, func):
        # Random perturbation gradient estimation
        perturbation = np.random.normal(0, self.perturbation_scale, self.dimension)
        perturbed_individual = np.clip(individual + perturbation, self.bounds[0], self.bounds[1])
        gradient = (
            (func(perturbed_individual) - func(individual))
            / (np.linalg.norm(perturbation) + 1e-16)
            * perturbation
        )
        return gradient

    def optimize(self, func):
        population = self.initialize_population()
        fitness = np.array([func(ind) for ind in population])
        best_individual, best_fitness = population[np.argmin(fitness)], np.min(fitness)

        evaluations = self.population_size

        while evaluations < self.budget:
            for i in range(self.population_size):
                gradient = self.approximate_gradient(population[i], func)
                new_individual = population[i] - gradient  # Gradient descent step
                new_individual = self.mutate(new_individual)  # Apply mutation
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
