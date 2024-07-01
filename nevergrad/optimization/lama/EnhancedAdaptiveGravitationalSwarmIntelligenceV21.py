import numpy as np


class EnhancedAdaptiveGravitationalSwarmIntelligenceV21:
    def __init__(
        self, budget=1000, G0=100.0, alpha_min=0.1, alpha_max=0.9, delta=0.1, gamma=0.1, eta=0.01, epsilon=0.1
    ):
        self.budget = budget
        self.G0 = G0
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.delta = delta
        self.gamma = gamma
        self.eta = eta
        self.epsilon = epsilon

    def initialize_population(self, population_size, func):
        return np.random.uniform(
            low=func.bounds.lb, high=func.bounds.ub, size=(population_size, func.bounds.lb.size)
        )

    def gravitational_force(self, x, xb, G):
        return G * (xb - x)

    def update_position(self, x, F, func):
        new_pos = x + F
        return np.clip(new_pos, func.bounds.lb, func.bounds.ub)

    def update_G(self, t):
        return self.G0 / (1.0 + self.eta * t)

    def update_alpha(self, t):
        return self.alpha_min + (self.alpha_max - self.alpha_min) * np.exp(-self.delta * t)

    def update_population(self, population, f_vals, func, G, best_pos):
        for i in range(len(population)):
            for j in range(len(population)):
                if np.random.rand() < self.alpha_max:
                    F = self.gravitational_force(population[i], population[j], G)
                    new_pos = self.update_position(population[i], F, func)
                    new_f_val = func(new_pos)

                    if new_f_val < f_vals[i]:
                        population[i] = new_pos
                        f_vals[i] = new_f_val

        return population, f_vals

    def update_parameters(self, t):
        self.G0 = self.update_G(t)
        self.alpha_max = self.update_alpha(t)

    def check_premature_convergence(self, f_vals):
        sorted_vals = np.sort(f_vals)
        diff = np.diff(sorted_vals)
        quartile = np.percentile(diff, 75)  # 75th percentile of the differences
        return quartile < self.epsilon

    def __call__(self, func):
        self.f_opt = np.Inf
        self.x_opt = None
        G = self.G0
        population_size = 20
        population = self.initialize_population(population_size, func)
        f_vals = np.array([func(x) for x in population])
        best_idx = np.argmin(f_vals)
        best_pos = population[best_idx]

        for t in range(self.budget):
            population, f_vals = self.update_population(population, f_vals, func, G, best_pos)

            for i in range(len(population)):
                if i != best_idx:
                    F = self.gravitational_force(population[i], best_pos, G)
                    new_pos = self.update_position(population[i], F, func)
                    new_f_val = func(new_pos)

                    if new_f_val < f_vals[i]:
                        population[i] = new_pos
                        f_vals[i] = new_f_val

            best_idx = np.argmin(f_vals)
            if f_vals[best_idx] < self.f_opt:
                self.f_opt = f_vals[best_idx]
                self.x_opt = population[best_idx]

            self.update_parameters(t)

            if self.check_premature_convergence(f_vals):
                break

        return self.f_opt, self.x_opt
