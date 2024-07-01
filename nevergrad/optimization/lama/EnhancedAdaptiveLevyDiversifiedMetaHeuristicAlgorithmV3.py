import numpy as np


class EnhancedAdaptiveLevyDiversifiedMetaHeuristicAlgorithmV3:
    def __init__(
        self,
        budget=10000,
        population_size=50,
        num_iterations=100,
        step_size=0.1,
        diversity_rate=0.2,
        levy_beta=1.8,
        alpha=0.5,
    ):
        self.budget = budget
        self.population_size = population_size
        self.dim = 5
        self.num_iterations = num_iterations
        self.step_size = step_size
        self.diversity_rate = diversity_rate
        self.levy_beta = levy_beta
        self.alpha = alpha

    def initialize_population(self):
        return np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))

    def levy_flight(self):
        sigma1 = (
            np.math.gamma(1 + self.levy_beta)
            * np.sin(np.pi * self.levy_beta / 2)
            / (np.math.gamma((1 + self.levy_beta) / 2) * self.levy_beta * 2 ** ((self.levy_beta - 1) / 2))
        ) ** (1 / self.levy_beta)
        sigma2 = 1
        u = np.random.normal(0, sigma1, self.dim)
        v = np.random.normal(0, sigma2, self.dim)
        levy = u / (np.abs(v) ** (1 / self.levy_beta))
        return levy

    def update_diversity_mutation(self, population):
        mask = np.random.rand(self.population_size, self.dim) < self.diversity_rate
        new_population = np.random.uniform(-5.0, 5.0, (self.population_size, self.dim))
        population = np.where(mask, new_population, population)
        return population

    def update_step_size(self, iter_count):
        return self.step_size / (1 + iter_count) ** self.alpha

    def __call__(self, func):
        population = self.initialize_population()
        fitness = np.array([func(sol) for sol in population])
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        best_fitness = fitness[best_idx]

        for i in range(self.budget // self.population_size):
            offspring_population = []
            for _ in range(self.population_size):
                new_solution = best_solution + self.step_size * self.levy_flight()
                offspring_population.append(new_solution)

            population = np.vstack((population, offspring_population))
            fitness = np.array([func(sol) for sol in population])
            sorted_indices = np.argsort(fitness)[: self.population_size]
            population = population[sorted_indices]
            fitness = np.array([func(sol) for sol in population])

            if fitness[0] < best_fitness:
                best_solution = population[0]
                best_fitness = fitness[0]

            population = self.update_diversity_mutation(population)
            self.step_size = self.update_step_size(i)

        return best_fitness, best_solution
