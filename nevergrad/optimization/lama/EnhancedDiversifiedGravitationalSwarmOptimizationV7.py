import numpy as np


class EnhancedDiversifiedGravitationalSwarmOptimizationV7:
    def __init__(
        self,
        budget=5000,
        G0=10.0,
        alpha=0.02,
        delta=0.005,
        gamma=0.1,
        population_size=300,
        rho_min=0.1,
        rho_max=0.35,
    ):
        self.budget = budget
        self.G0 = G0
        self.alpha = alpha
        self.delta = delta
        self.gamma = gamma
        self.population_size = population_size
        self.rho_min = rho_min
        self.rho_max = rho_max

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
        return self.G0 / (1.0 + self.alpha * t)

    def update_alpha(self, t):
        return self.alpha * np.exp(-self.delta * t)

    def evolve_population(self, population, f_vals, func):
        G = self.G0
        best_idx = np.argmin(f_vals)
        best_pos = population[best_idx]
        best_val = f_vals[best_idx]

        for t in range(self.budget):
            rho = self.rho_min + (self.rho_max - self.rho_min) * (1 - t / self.budget)

            for i in range(self.population_size):
                j = np.random.choice(range(self.population_size))
                F = self.gravitational_force(population[i], population[j], G)
                new_pos = self.update_position(population[i], F, func)
                new_f_val = func(new_pos)

                if new_f_val < f_vals[i]:
                    population[i] = new_pos
                    f_vals[i] = new_f_val

                if new_f_val < best_val:
                    best_pos = new_pos
                    best_val = new_f_val

            G = self.update_G(t)
            self.alpha = self.update_alpha(t)

            for i in range(self.population_size):
                if np.random.rand() < rho:
                    population[i] = np.random.uniform(func.bounds.lb, func.bounds.ub)
                    f_vals[i] = func(population[i])

        return best_val, best_pos

    def __call__(self, func):
        best_aooc = np.Inf
        best_x_opt = None

        for _ in range(300):  # Increase the number of runs
            population = self.initialize_population(func)
            f_vals = np.array([func(x) for x in population])

            for _ in range(150):  # Increase the number of iterations within each run
                best_f_val, best_pos = self.evolve_population(population, f_vals, func)

            if best_f_val < best_aooc:
                best_aooc = best_f_val
                best_x_opt = best_pos

        return best_aooc, best_x_opt
