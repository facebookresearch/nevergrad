import numpy as np


class EnhancedGravitationalSwarmIntelligenceV14:
    def __init__(
        self,
        budget=1000,
        G0=100.0,
        alpha_min=0.1,
        alpha_max=0.9,
        delta=0.1,
        gamma=0.1,
        eta=0.01,
        epsilon=0.1,
        population_size=50,
    ):
        self.budget = budget
        self.G0 = G0
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.delta = delta
        self.gamma = gamma
        self.eta = eta
        self.epsilon = epsilon
        self.population_size = population_size

    def initialize_population(self, func):
        return np.random.uniform(
            low=func.bounds.lb, high=func.bounds.ub, size=(self.population_size, len(func.bounds.lb))
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

    def update_population(self, population, f_vals, func, G, best_pos, t):
        alpha_t = self.update_alpha(t)
        for i in range(len(population)):
            for j in range(len(population)):
                if np.random.rand() < alpha_t:
                    F = self.gravitational_force(population[i], population[j], G)
                    new_pos = self.update_position(population[i], F, func)
                    new_f_val = func(new_pos)

                    if new_f_val < f_vals[i]:
                        population[i] = new_pos
                        f_vals[i] = new_f_val

        return population, f_vals

    def update_parameters(self, t, f_vals):
        avg_f = np.mean(f_vals)
        self.G0 = self.update_G(t)
        self.alpha_max = self.update_alpha(t)
        self.delta = 1 / (1 + np.exp(-self.gamma * (avg_f - np.min(f_vals) + self.epsilon)))

    def evolve_population(self, population, f_vals, func):
        G = self.G0
        best_idx = np.argmin(f_vals)
        best_pos = population[best_idx]
        best_val = f_vals[best_idx]

        for t in range(self.budget):
            population, f_vals = self.update_population(population, f_vals, func, G, best_pos, t)
            self.update_parameters(t, f_vals)

            for i in range(len(population)):
                F = self.gravitational_force(population[i], best_pos, G)
                new_pos = self.update_position(population[i], F, func)
                new_f_val = func(new_pos)

                if new_f_val < f_vals[i]:
                    population[i] = new_pos
                    f_vals[i] = new_f_val

                if new_f_val < best_val:
                    best_pos = new_pos
                    best_val = new_f_val

        return best_val, best_pos

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None
        population = self.initialize_population(func)
        f_vals = np.array([func(x) for x in population])

        for _ in range(5):  # Increase the number of optimization runs to 5
            self.f_opt, self.x_opt = self.evolve_population(population, f_vals, func)
            if self.f_opt < np.inf:  # If a valid solution is found, break
                break

        return self.f_opt, self.x_opt
