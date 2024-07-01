import numpy as np


class HybridAdaptiveCovarianceMatrixDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.dim = 5
        self.population_size = 100  # Increased population size
        self.sigma = 0.2  # Step size
        self.c1 = 0.05  # Learning rate for rank-one update
        self.cmu = 0.01  # Learning rate for rank-mu update
        self.damping = 1 + (self.dim / (2 * self.population_size))
        self.weights = np.log(self.population_size / 2 + 1) - np.log(
            np.arange(1, self.population_size // 2 + 1)
        )
        self.weights /= np.sum(self.weights)
        self.mu = len(self.weights)
        self.adaptive_learning_rate = 0.02  # Further fine-tuned learning rate for mutation adaptability
        self.elitism_rate = 0.1
        self.eval_count = 0
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability

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

        def differential_evolution(population, fitness):
            new_population = np.copy(population)
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = population[indices[0]], population[indices[1]], population[indices[2]]
                mutant_vector = clip_bounds(x1 + self.F * (x2 - x3))
                crossover = np.random.rand(self.dim) < self.CR
                if not np.any(crossover):
                    crossover[np.random.randint(self.dim)] = True
                trial_vector = np.where(crossover, mutant_vector, population[i])
                trial_vector = clip_bounds(trial_vector)
                trial_fitness = func(trial_vector)
                self.eval_count += 1
                if trial_fitness < fitness[i]:
                    new_population[i] = trial_vector
                    fitness[i] = trial_fitness
            return new_population, fitness

        def retain_elite(population, fitness, new_population, new_fitness):
            combined_population = np.vstack((population, new_population))
            combined_fitness = np.hstack((fitness, new_fitness))
            sorted_indices = np.argsort(combined_fitness)
            elite_count = int(self.elitism_rate * self.population_size)
            retained_indices = sorted_indices[: self.population_size - elite_count]
            retained_population = combined_population[retained_indices]
            retained_fitness = combined_fitness[retained_indices]
            elite_indices = sorted_indices[:elite_count]
            elite_population = combined_population[elite_indices]
            elite_fitness = combined_fitness[elite_indices]
            return np.vstack((retained_population, elite_population)), np.hstack(
                (retained_fitness, elite_fitness)
            )

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

            new_population, new_fitness = differential_evolution(offspring, fitness.copy())

            population, fitness = retain_elite(population, fitness, new_population, new_fitness)

            best_index = np.argmin(fitness)
            if fitness[best_index] < best_value:
                best_value = fitness[best_index]
                best_position = population[best_index]

        return best_value, best_position


# Example usage:
# func = SomeBlackBoxFunction()  # The black box function to be optimized
# optimizer = HybridAdaptiveCovarianceMatrixDifferentialEvolution(budget=10000)
# best_value, best_position = optimizer(func)
