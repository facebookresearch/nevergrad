import numpy as np


class EvolutionaryHarmonicFireworkAlgorithm:
    def __init__(self, budget=10000, n_fireworks=30, n_sparks=10, scale_factor=0.1):
        self.budget = budget
        self.n_fireworks = n_fireworks
        self.n_sparks = n_sparks
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.f_opt = np.inf
        self.x_opt = None
        self.scale_factor = scale_factor

    def initialize_fireworks(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.n_fireworks, self.dim))

    def explode_firework(self, firework):
        sparks = np.random.uniform(
            firework - self.scale_factor, firework + self.scale_factor, (self.n_sparks, self.dim)
        )
        return sparks

    def levy_flight(self, step_size=0.1):
        beta = 1.5
        u = np.random.normal(0, 1, size=self.dim)
        v = np.random.normal(0, 1, size=self.dim)
        step = u / abs(v) ** (1 / beta)
        return step_size * step

    def clip_to_bounds(self, x):
        return np.clip(x, self.bounds[0], self.bounds[1])

    def evolve_fireworks(self, fireworks, func):
        for i in range(self.n_fireworks):
            sparks = self.explode_firework(fireworks[i])

            for spark in sparks:
                if func(spark) < func(fireworks[i]):
                    fireworks[i] = spark

            for _ in range(self.n_sparks):
                idx1, idx2 = np.random.choice(self.n_fireworks, 2, replace=False)
                trial = fireworks[i] + 0.5 * (fireworks[idx1] - fireworks[idx2])
                trial = self.clip_to_bounds(trial)
                if func(trial) < func(fireworks[i]):
                    fireworks[i] = trial

        return fireworks

    def update_best_firework(self, fireworks, func):
        best_idx = np.argmin([func(firework) for firework in fireworks])
        if func(fireworks[best_idx]) < self.f_opt:
            self.f_opt = func(fireworks[best_idx])
            self.x_opt = fireworks[best_idx]

    def __call__(self, func):
        fireworks = self.initialize_fireworks()

        for _ in range(self.budget):
            fireworks = self.evolve_fireworks(fireworks, func)

        self.update_best_firework(fireworks, func)

        return self.f_opt, self.x_opt
