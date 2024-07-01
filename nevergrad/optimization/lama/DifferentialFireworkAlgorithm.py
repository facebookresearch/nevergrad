import numpy as np


class DifferentialFireworkAlgorithm:
    def __init__(self, budget=10000, n_fireworks=20, n_sparks=10, scaling_factor=0.5, crossover_rate=0.9):
        self.budget = budget
        self.n_fireworks = n_fireworks
        self.n_sparks = n_sparks
        self.scaling_factor = scaling_factor
        self.crossover_rate = crossover_rate
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

    def __call__(self, func):
        fireworks = self.initialize_fireworks()

        for _ in range(int(self.budget / self.n_fireworks)):
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

        return self.f_opt, self.x_opt
