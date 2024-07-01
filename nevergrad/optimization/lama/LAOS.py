import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


class LAOS:
    def __init__(self, budget):
        self.budget = budget
        self.dimension = 5
        self.bounds = np.array([-5.0, 5.0])

    def initialize(self):
        population_size = 30
        population = np.random.uniform(*self.bounds, (population_size, self.dimension))
        return population, population_size

    def evaluate(self, population, func):
        return np.array([func(ind) for ind in population])

    def select_best(self, population, fitness):
        best_idx = np.argmin(fitness)
        return population[best_idx], fitness[best_idx]

    def layered_search_process(self, initial_population, func):
        layer_depth = 3
        current_population = initial_population
        evaluations = len(current_population)
        for layer in range(layer_depth):
            current_fitness = self.evaluate(current_population, func)
            evaluations += len(current_population)
            if evaluations >= self.budget:
                break

            # Learn landscape features using Gaussian Process
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
            model = GaussianProcessRegressor(kernel=kernel)
            model.fit(current_population, current_fitness)

            # Predict and refine search around best solutions
            best_individuals_idx = np.argsort(current_fitness)[: max(5, len(current_fitness) // 10)]
            best_individuals = current_population[best_individuals_idx]
            new_population = []
            for best in best_individuals:
                perturbation = np.random.normal(0, 0.1 / (layer + 1), (5, self.dimension))
                new_candidates = best + perturbation
                new_population.append(new_candidates)
            current_population = np.vstack(new_population)
            current_population = np.clip(current_population, *self.bounds)

        return self.select_best(current_population, self.evaluate(current_population, func))

    def __call__(self, func):
        initial_population, _ = self.initialize()
        best_solution, best_fitness = self.layered_search_process(initial_population, func)
        return best_fitness, best_solution
