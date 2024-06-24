import numpy as np


class EnhancedCMAES:
    def __init__(self, budget=10000, mu=None, sigma=1):
        self.budget = budget
        self.mu = mu if mu is not None else 20
        self.dim = 5
        self.sigma = sigma
        self.population = np.random.uniform(-5.0, 5.0, size=(self.mu, self.dim))
        self.weights = np.log(self.mu + 1 / 2) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mu_eff = 1 / np.sum(self.weights**2)
        self.cov_matrix = np.eye(self.dim)
        self.best_fitness = np.Inf
        self.best_solution = None

    def sample_population(self):
        return np.random.multivariate_normal(np.zeros(self.dim), self.cov_matrix, self.mu)

    def evaluate_population(self, func, population):
        fitness = np.array([func(sol) for sol in population])
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < self.best_fitness:
            self.best_fitness = fitness[min_idx]
            self.best_solution = population[min_idx].copy()
        return fitness

    def update_parameters(self, population, fitness):
        z = (population - np.mean(population, axis=0)) / np.sqrt(self.sigma**2)
        rank = np.argsort(fitness)
        y = np.sum(self.weights[:, None] * z[rank[: self.mu]], axis=0)
        self.cov_matrix = np.cov(population, rowvar=False, aweights=self.weights, bias=True)

        self.sigma *= np.exp((np.linalg.norm(y) - self.mu_eff) / (2 * np.sqrt(self.dim)))

    def __call__(self, func):
        for _ in range(self.budget):
            children = self.sample_population()
            fitness = self.evaluate_population(func, children)
            self.update_parameters(children, fitness)

        aocc = 1 - np.std(fitness) / np.mean(fitness)
        return aocc, self.best_solution
