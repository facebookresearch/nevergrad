import numpy as np


class AdaptiveCovarianceMatrixEvolutionWithSelfAdaptiveMutation:
    def __init__(self, budget=10000):
        self.budget = budget
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.dim = 5
        self.population_size = 100  # Population size
        self.sigma = 0.3  # Initial step size
        self.c1 = 0.01  # Learning rate for rank-one update
        self.cmu = 0.01  # Learning rate for rank-mu update
        self.damping = 1 + (self.dim / (2 * self.population_size))  # Damping factor for step size
        self.weights = np.log(self.population_size / 2 + 1) - np.log(
            np.arange(1, self.population_size // 2 + 1)
        )
        self.weights /= np.sum(self.weights)
        self.mu = len(self.weights)  # Number of parents for recombination
        self.adaptive_learning_rate = 0.1  # Learning rate for adaptive self-adaptive mutation
        self.eval_count = 0

    def __call__(self, func):
        def clip_bounds(candidate):
            return np.clip(candidate, self.lower_bound, self.upper_bound)

        def initialize_population():
            population = np.random.uniform(
                self.lower_bound, self.upper_bound, (self.population_size, self.dim)
            )
            fitness = np.array([func(ind) for ind in population])
            return population, fitness

        def adapt_sigma():
            self.sigma *= np.exp(self.adaptive_learning_rate * (np.random.randn() - 0.5))

        def recombination(population, fitness):
            sorted_indices = np.argsort(fitness)
            selected_population = population[sorted_indices[: self.mu]]
            recombined = np.dot(self.weights, selected_population)
            return recombined, sorted_indices, selected_population

        def update_covariance_matrix(cov_matrix, selected_population, mean, recombined):
            z = (selected_population - mean) / self.sigma
            rank_one = np.outer(z[0], z[0])
            rank_mu = sum(self.weights[i] * np.outer(z[i], z[i]) for i in range(self.mu))
            cov_matrix = (1 - self.c1 - self.cmu) * cov_matrix + self.c1 * rank_one + self.cmu * rank_mu
            return cov_matrix

        def sample_offspring(recombined, cov_matrix):
            offspring = np.random.multivariate_normal(
                recombined, self.sigma**2 * cov_matrix, self.population_size
            )
            return clip_bounds(offspring)

        population, fitness = initialize_population()
        cov_matrix = np.identity(self.dim)

        best_index = np.argmin(fitness)
        best_position = population[best_index]
        best_value = fitness[best_index]

        mean = np.mean(population, axis=0)

        while self.eval_count < self.budget:
            adapt_sigma()
            recombined, sorted_indices, selected_population = recombination(population, fitness)
            cov_matrix = update_covariance_matrix(cov_matrix, selected_population, mean, recombined)
            offspring = sample_offspring(recombined, cov_matrix)

            new_fitness = np.array([func(ind) for ind in offspring])
            self.eval_count += self.population_size

            population = offspring
            fitness = new_fitness

            best_index = np.argmin(fitness)
            if fitness[best_index] < best_value:
                best_value = fitness[best_index]
                best_position = population[best_index]

        return best_value, best_position


# Example usage:
# func = SomeBlackBoxFunction()  # The black box function to be optimized
# optimizer = AdaptiveCovarianceMatrixEvolutionWithSelfAdaptiveMutation(budget=10000)
# best_value, best_position = optimizer(func)
