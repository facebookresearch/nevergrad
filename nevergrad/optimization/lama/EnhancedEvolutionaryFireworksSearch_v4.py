import numpy as np


class EnhancedEvolutionaryFireworksSearch_v4:
    def __init__(self, budget=10000, n_fireworks=20, n_sparks=10, alpha=0.1, beta=1.5, gamma=1.0, delta=0.2):
        self.budget = budget
        self.n_fireworks = n_fireworks
        self.n_sparks = n_sparks
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.f_opt = np.inf
        self.x_opt = None

    def initialize_fireworks(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.n_fireworks, self.dim))

    def explode_firework(self, firework):
        sparks = np.random.uniform(firework - self.alpha, firework + self.alpha, (self.n_sparks, self.dim))
        return sparks

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

    def clip_to_bounds(self, x):
        return np.clip(x, self.bounds[0], self.bounds[1])

    def enhance_fireworks(self, fireworks):
        for i in range(self.n_fireworks):
            for j in range(self.dim):
                fireworks[i][j] += self.levy_flight() * np.random.normal(0, 1)
                fireworks[i][j] = self.clip_to_bounds(fireworks[i][j])
        return fireworks

    def evolve_fireworks(self, fireworks, func):
        for i in range(self.n_fireworks):
            sparks = self.explode_firework(fireworks[i])

            for spark in sparks:
                if func(spark) < func(fireworks[i]):
                    fireworks[i] = spark

            for j in range(self.n_sparks):
                idx1, idx2 = np.random.choice(np.delete(np.arange(self.n_fireworks), i), 2, replace=False)
                trial = fireworks[i] + self.beta * (fireworks[idx1] - fireworks[idx2])
                trial = self.clip_to_bounds(trial)
                if func(trial) < func(fireworks[i]):
                    fireworks[i] = trial

        return fireworks

    def adapt_parameters(self, it):
        self.beta = max(1.0, self.beta * 0.995)
        return self.beta

    def __call__(self, func):
        fireworks = self.initialize_fireworks()

        for it in range(self.budget):
            fireworks = self.enhance_fireworks(fireworks)
            fireworks = self.evolve_fireworks(fireworks, func)
            self.beta = self.adapt_parameters(it)

            best_idx = np.argmin([func(firework) for firework in fireworks])
            if func(fireworks[best_idx]) < self.f_opt:
                self.f_opt = func(fireworks[best_idx])
                self.x_opt = fireworks[best_idx]

        return self.f_opt, self.x_opt
