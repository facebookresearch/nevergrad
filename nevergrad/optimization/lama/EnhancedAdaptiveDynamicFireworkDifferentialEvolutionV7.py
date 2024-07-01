import numpy as np


class EnhancedAdaptiveDynamicFireworkDifferentialEvolutionV7:
    def __init__(
        self, budget=10000, n_fireworks=50, n_sparks=15, f_init=0.8, f_final=0.2, cr_init=0.9, cr_final=0.1
    ):
        self.budget = budget
        self.n_fireworks = n_fireworks
        self.n_sparks = n_sparks
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.f_opt = np.inf
        self.x_opt = None
        self.f_init = f_init
        self.f_final = f_final
        self.cr_init = cr_init
        self.cr_final = cr_final

    def initialize_fireworks(self):
        return np.random.uniform(self.bounds[0], self.bounds[1], (self.n_fireworks, self.dim))

    def explode_firework(self, firework, alpha):
        sparks = np.random.uniform(firework - alpha, firework + alpha, (self.n_sparks, self.dim))
        return sparks

    def clip_to_bounds(self, x):
        return np.clip(x, self.bounds[0], self.bounds[1])

    def evolve_fireworks(self, fireworks, func, f, cr):
        for i in range(self.n_fireworks):
            alpha = 0.5 * (1 - i / self.n_fireworks)  # Dynamic alpha based on iteration
            sparks = self.explode_firework(fireworks[i], alpha)

            for _ in range(self.n_sparks):
                idx1, idx2, idx3 = np.random.choice(self.n_fireworks, 3, replace=False)
                mutant = self.clip_to_bounds(fireworks[idx1] + f * (fireworks[idx2] - fireworks[idx3]))

                trial = np.where(np.random.rand(self.dim) < cr, mutant, fireworks[i])
                if func(trial) < func(fireworks[i]):
                    fireworks[i] = trial

        return fireworks

    def update_best_firework(self, fireworks, func):
        best_idx = np.argmin([func(firework) for firework in fireworks])
        if func(fireworks[best_idx]) < self.f_opt:
            self.f_opt = func(fireworks[best_idx])
            self.x_opt = fireworks[best_idx]

    def adapt_params(self, iteration):
        f = self.f_init + (self.f_final - self.f_init) * (iteration / self.budget) ** 0.5
        cr = self.cr_init + (self.cr_final - self.cr_init) * (iteration / self.budget) ** 0.5
        return f, cr

    def __call__(self, func):
        fireworks = self.initialize_fireworks()

        for i in range(self.budget):
            f, cr = self.adapt_params(i)
            fireworks = self.evolve_fireworks(fireworks, func, f, cr)

        self.update_best_firework(fireworks, func)

        return self.f_opt, self.x_opt
