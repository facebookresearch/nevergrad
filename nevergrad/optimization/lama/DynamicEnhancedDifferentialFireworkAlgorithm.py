import numpy as np


class DynamicEnhancedDifferentialFireworkAlgorithm:
    def __init__(
        self,
        budget=10000,
        n_fireworks=20,
        n_sparks=10,
        scaling_factor=0.5,
        crossover_rate=0.9,
        levy_flight_prob=0.3,
    ):
        self.budget = budget
        self.n_fireworks = n_fireworks
        self.n_sparks = n_sparks
        self.scaling_factor = scaling_factor
        self.crossover_rate = crossover_rate
        self.levy_flight_prob = levy_flight_prob
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.f_opt = np.inf
        self.x_opt = None

    def initialize_fireworks(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.n_fireworks, self.dim))

    def explode_firework(self, firework):
        sparks = np.random.uniform(
            firework - self.scaling_factor, firework + self.scaling_factor, (self.n_sparks, self.dim)
        )
        return sparks

    def differential_evolution(self, current, target1, target2):
        mutant = current + self.scaling_factor * (target1 - target2)
        crossover_points = np.random.rand(self.dim) < self.crossover_rate
        trial = np.where(crossover_points, mutant, current)
        return trial

    def clip_to_bounds(self, x):
        return np.clip(x, self.bounds[0], self.bounds[1])

    def levy_flight(self, step_size=0.1):
        beta = 1.5
        sigma = (
            np.math.gamma(1 + beta)
            * np.math.sin(np.pi * beta / 2)
            / (np.math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
        ) ** (1 / beta)
        u = np.random.normal(0, sigma)
        v = np.random.normal(0, 1)
        step = u / abs(v) ** (1 / beta)
        return step_size * step

    def enhance_fireworks(self, fireworks):
        for i in range(self.n_fireworks):
            if np.random.rand() < self.levy_flight_prob:
                fireworks[i] += self.levy_flight() * (fireworks[i] - self.x_opt)
                fireworks[i] = self.clip_to_bounds(fireworks[i])
        return fireworks

    def adjust_parameters(self, iteration):
        self.scaling_factor = 0.5 - 0.4 * (iteration / self.budget)
        self.levy_flight_prob = 0.3 - 0.25 * (iteration / self.budget)

    def __call__(self, func):
        fireworks = self.initialize_fireworks()

        for it in range(self.budget):
            self.adjust_parameters(it)

            for i in range(self.n_fireworks):
                sparks = self.explode_firework(fireworks[i])
                for j in range(self.n_sparks):
                    idx1, idx2 = np.random.choice(np.delete(np.arange(self.n_fireworks), i), 2, replace=False)
                    trial = self.differential_evolution(fireworks[i], fireworks[idx1], fireworks[idx2])
                    trial = self.clip_to_bounds(trial)
                    if func(trial) < func(fireworks[i]):
                        fireworks[i] = trial

            best_idx = np.argmin([func(firework) for firework in fireworks])
            if func(fireworks[best_idx]) < self.f_opt:
                self.f_opt = func(fireworks[best_idx])
                self.x_opt = fireworks[best_idx]

            fireworks = self.enhance_fireworks(fireworks)

        return self.f_opt, self.x_opt
