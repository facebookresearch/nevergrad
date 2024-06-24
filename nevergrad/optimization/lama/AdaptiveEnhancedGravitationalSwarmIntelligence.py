import numpy as np


class AdaptiveEnhancedGravitationalSwarmIntelligence:
    def __init__(
        self,
        budget=1000,
        population_size=20,
        G0=100.0,
        alpha=0.1,
        beta_min=0.1,
        beta_max=0.9,
        delta=0.1,
        gamma=0.1,
        eta=0.1,
    ):
        self.budget = budget
        self.population_size = population_size
        self.G0 = G0
        self.alpha = alpha
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.delta = delta
        self.gamma = gamma
        self.eta = eta

    def initialize_population(self, func):
        return np.random.uniform(
            low=func.bounds.lb, high=func.bounds.ub, size=(self.population_size, func.bounds.lb.size)
        )

    def gravitational_force(self, x, xb, G):
        return G * (xb - x)

    def update_position(self, x, F):
        return x + F

    def update_G(self, t):
        return self.G0 * np.exp(-self.alpha * t)

    def update_beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * np.exp(-self.gamma * t)

    def update_alpha(self, t):
        return self.alpha * (1.0 - self.delta)

    def update_population(self, population, f_vals, func, G):
        for i in range(self.population_size):
            if np.random.rand() < self.beta_max:
                random_index = np.random.choice([idx for idx in range(self.population_size) if idx != i])
                F = self.gravitational_force(population[i], population[random_index], G)
                new_pos = self.update_position(population[i], F)
                new_pos = np.clip(new_pos, func.bounds.lb, func.bounds.ub)
                new_f_val = func(new_pos)

                if new_f_val < f_vals[i]:
                    population[i] = new_pos
                    f_vals[i] = new_f_val

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
            population, f_vals = self.update_population(population, f_vals, func, G)

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

            G = self.update_G(t)
            self.alpha = self.update_alpha(t)
            self.beta_max = self.update_beta(t)

        return self.f_opt, self.x_opt
