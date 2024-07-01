import numpy as np


class AdvancedGradientEvolutionStrategyV2:
    def __init__(
        self, budget, dim=5, pop_size=100, tau=0.3, sigma_init=0.5, learning_rate=0.02, grad_approx_steps=10
    ):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.tau = tau  # Enhanced learning rate for step size adaptation
        self.sigma_init = sigma_init  # Adjusted initial step size
        self.learning_rate = learning_rate  # Adjusted gradient descent learning rate
        self.grad_approx_steps = grad_approx_steps  # Steps to approximate gradient
        self.bounds = (-5.0, 5.0)

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], size=(self.pop_size, self.dim))

    def mutate(self, individual, sigma):
        return np.clip(individual + sigma * np.random.randn(self.dim), self.bounds[0], self.bounds[1])

    def estimate_gradient(self, func, individual):
        grad = np.zeros(self.dim)
        f_base = func(individual)
        for i in range(self.dim):
            perturb = np.zeros(self.dim)
            eps = self.sigma_init / np.sqrt(self.dim)
            perturb[i] = eps
            f_plus = func(individual + perturb)
            grad[i] = (f_plus - f_base) / eps
        return grad

    def __call__(self, func):
        population = self.initialize_population()
        f_values = np.array([func(ind) for ind in population])
        n_evals = self.pop_size
        sigma_values = np.full(self.pop_size, self.sigma_init)

        while n_evals < self.budget:
            new_population = []
            new_f_values = []

            for idx in range(self.pop_size):
                individual = population[idx]
                sigma = sigma_values[idx]
                gradient = np.zeros(self.dim)

                for step in range(self.grad_approx_steps):
                    gradient += self.estimate_gradient(func, individual)
                gradient /= self.grad_approx_steps

                individual_new = np.clip(
                    individual - self.learning_rate * gradient, self.bounds[0], self.bounds[1]
                )
                f_new = func(individual_new)
                n_evals += 1

                new_population.append(individual_new)
                new_f_values.append(f_new)

                if n_evals >= self.budget:
                    break

            population = np.array(new_population)
            f_values = np.array(new_f_values)
            rankings = np.argsort(f_values)
            best_sigma = sigma_values[rankings[0]]
            sigma_values = best_sigma * np.exp(self.tau * (np.random.randn(self.pop_size) - 0.5))

        self.f_opt = np.min(f_values)
        self.x_opt = population[np.argmin(f_values)]

        return self.f_opt, self.x_opt
