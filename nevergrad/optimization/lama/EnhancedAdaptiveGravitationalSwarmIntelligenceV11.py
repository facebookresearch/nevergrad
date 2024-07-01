import numpy as np


class EnhancedAdaptiveGravitationalSwarmIntelligenceV11:
    def __init__(
        self,
        budget=1000,
        population_size=20,
        G0=100.0,
        alpha_min=0.1,
        alpha_max=0.9,
        beta_min=0.1,
        beta_max=0.9,
        delta=0.1,
        gamma=0.1,
        eta=0.01,
        epsilon=0.1,
        mu=0.1,
    ):
        self.budget = budget
        self.population_size = population_size
        self.G0 = G0
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.delta = delta
        self.gamma = gamma
        self.eta = eta
        self.epsilon = epsilon
        self.mu = mu

    def initialize_population(self, func):
        return np.random.uniform(
            low=func.bounds.lb, high=func.bounds.ub, size=(self.population_size, func.bounds.lb.size)
        )

    def gravitational_force(self, x, xb, G):
        return G * (xb - x)

    def update_position(self, x, F):
        return x + F

    def update_G(self, t):
        return self.G0 / (1.0 + self.eta * t)

    def update_beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * np.exp(-self.gamma * t)

    def update_alpha(self, t):
        return self.alpha_min + (self.alpha_max - self.alpha_min) * np.exp(-self.delta * t)

    def update_population(self, population, f_vals, func, G, best_pos):
        for i in range(self.population_size):
            for j in range(self.population_size):
                if np.random.rand() < self.beta_max:
                    F = self.gravitational_force(population[i], population[j], G)
                    new_pos = self.update_position(population[i], F)
                    new_pos = np.clip(new_pos, func.bounds.lb, func.bounds.ub)
                    new_f_val = func(new_pos)

                    if new_f_val < f_vals[i]:
                        population[i] = new_pos
                        f_vals[i] = new_f_val

        return population, f_vals

    def update_parameters(self, t):
        self.G0 = self.update_G(t)
        self.beta_max = self.update_beta(t)
        self.alpha_min = self.update_alpha(t)

    def perturb_population(self, func, population, f_vals):
        for i in range(self.population_size):
            if np.random.rand() < self.mu:
                population[i] = np.random.uniform(low=func.bounds.lb, high=func.bounds.ub)
                f_vals[i] = func(population[i])
        return population, f_vals

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        G = self.G0
        population = self.initialize_population(func)
        f_vals = np.array([func(x) for x in population])
        best_idx = np.argmin(f_vals)
        best_pos = population[best_idx]

        for t in range(self.budget):
            population, f_vals = self.update_population(population, f_vals, func, G, best_pos)

            for i in range(self.population_size):
                if i != best_idx:
                    F = self.gravitational_force(population[i], best_pos, G)
                    new_pos = self.update_position(population[i], F)
                    new_pos = np.clip(new_pos, func.bounds.lb, func.bounds.ub)
                    new_f_val = func(new_pos)

                    if new_f_val < f_vals[i]:
                        population[i] = new_pos
                        f_vals[i] = new_f_val

            best_idx = np.argmin(f_vals)
            if f_vals[best_idx] < self.f_opt:
                self.f_opt = f_vals[best_idx]
                self.x_opt = population[best_idx]

            self.update_parameters(t)

            population, f_vals = self.perturb_population(func, population, f_vals)

        return self.f_opt, self.x_opt
