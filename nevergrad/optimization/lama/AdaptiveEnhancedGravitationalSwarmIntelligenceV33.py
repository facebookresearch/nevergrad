import numpy as np


class AdaptiveEnhancedGravitationalSwarmIntelligenceV33:
    def __init__(
        self,
        budget=3000,
        G0=100.0,
        alpha_min=0.1,
        alpha_max=0.9,
        delta=0.1,
        gamma=0.1,
        eta=0.01,
        epsilon=0.1,
        population_size=300,
        elite_percentage=0.1,
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
        self.elite_size = int(elite_percentage * population_size)

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

            self.update_parameters(t, f_vals)

        return best_val, best_pos

    def __call__(self, func):
        best_aooc = np.Inf
        best_x_opt = None
        best_std = np.Inf

        for _ in range(300):  # Increased the number of optimization runs to 300
            population = self.initialize_population(func)
            f_vals = np.array([func(x) for x in population])

            for _ in range(4000):  # Increased the number of iterations within each optimization run to 4000
                best_f_val, best_pos = self.evolve_population(population, f_vals, func)

            # Update the best AOCC, optimal solution, and standard deviation
            if best_f_val < best_aooc:
                best_aooc = best_f_val
                best_x_opt = best_pos
                best_std = np.std(f_vals)

        return best_aooc, best_x_opt, best_std
