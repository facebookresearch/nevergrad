import numpy as np


class GradientGuidedEvolutionStrategy:
    def __init__(self, budget, dim=5, pop_size=50, tau=0.2, sigma=0.1, beta=0.07):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.tau = tau  # Learning rate for step size adaptation
        self.sigma = sigma  # Initial step size
        self.beta = beta  # Gradient estimation perturbation magnitude
        self.bounds = (-5.0, 5.0)

    def initialize_population(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], size=(self.pop_size, self.dim))

    def mutate(self, individual, sigma):
        return np.clip(individual + sigma * np.random.randn(self.dim), self.bounds[0], self.bounds[1])

    def estimate_gradient(self, func, individual, sigma):
        grad = np.zeros(self.dim)
        for i in range(self.dim):
            perturb = np.zeros(self.dim)
            perturb[i] = self.beta * sigma
            f_plus = func(individual + perturb)
            f_minus = func(individual - perturb)
            grad[i] = (f_plus - f_minus) / (2 * self.beta * sigma)
        return grad

    def __call__(self, func):
        population = self.initialize_population()
        f_values = np.array([func(ind) for ind in population])
        n_evals = self.pop_size
        sigma_values = np.full(self.pop_size, self.sigma)

        while n_evals < self.budget:
            for idx in range(self.pop_size):
                individual = population[idx]
                sigma = sigma_values[idx]
                grad = self.estimate_gradient(func, individual, sigma)
                individual_new = np.clip(individual - sigma * grad, self.bounds[0], self.bounds[1])
                f_new = func(individual_new)
                n_evals += 1
                if f_new < f_values[idx]:
                    population[idx] = individual_new
                    f_values[idx] = f_new
                    sigma_values[idx] *= np.exp(self.tau * np.linalg.norm(grad))
                else:
                    sigma_values[idx] *= np.exp(-self.tau)

                if n_evals >= self.budget:
                    break

        self.f_opt = np.min(f_values)
        self.x_opt = population[np.argmin(f_values)]

        return self.f_opt, self.x_opt
