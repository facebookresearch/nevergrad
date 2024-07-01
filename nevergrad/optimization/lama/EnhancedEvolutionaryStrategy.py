import numpy as np


class EnhancedEvolutionaryStrategy:
    def __init__(self, budget=10000, mu=10, lambda_=20, tau=1 / np.sqrt(2), tau_prime=1 / np.sqrt(2)):
        self.budget = budget
        self.mu = mu
        self.lambda_ = lambda_
        self.tau = tau
        self.tau_prime = tau_prime
        self.dim = 5
        self.population = np.random.uniform(-5.0, 5.0, size=(self.mu, self.dim))
        self.best_fitness = np.Inf
        self.best_solution = None

    def mutate_population(self, population, tau, tau_prime):
        mutated_population = population + np.random.normal(0, tau, size=population.shape)
        mutated_population += np.random.normal(0, tau_prime, size=population.shape)
        return np.clip(mutated_population, -5.0, 5.0)

    def evaluate_population(self, func, population):
        fitness = np.array([func(sol) for sol in population])
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < self.best_fitness:
            self.best_fitness = fitness[min_idx]
            self.best_solution = population[min_idx].copy()
        return fitness

    def selection(self, population, fitness, mu):
        idx = np.argsort(fitness)[:mu]
        return population[idx].copy()

    def __call__(self, func):
        tau = self.tau
        tau_prime = self.tau_prime
        for _ in range(self.budget):
            children = self.mutate_population(self.population, tau, tau_prime)
            fitness = self.evaluate_population(func, children)
            self.population = self.selection(children, fitness, self.mu)

            # Adapt strategy parameters
            tau *= np.exp((1 / np.sqrt(2 * self.dim)) * np.random.normal(0, 1))
            tau_prime *= np.exp((1 / np.sqrt(2 * np.sqrt(self.dim))) * np.random.normal(0, 1))

        aocc = 1 - np.std(fitness) / np.mean(fitness)
        return aocc, self.best_solution
