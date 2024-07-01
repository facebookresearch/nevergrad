import numpy as np


class DynamicFireworkAlgorithm:
    def __init__(self, budget=10000, n_fireworks=10, n_sparks=5, alpha=0.5, beta=2.0, mutation_rate=0.1):
        self.budget = budget
        self.n_fireworks = n_fireworks
        self.n_sparks = n_sparks
        self.alpha = alpha
        self.beta = beta
        self.mutation_rate = mutation_rate
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.f_opt = np.inf
        self.x_opt = None

    def initialize_fireworks(self, func):
        self.fireworks = np.random.uniform(self.bounds[0], self.bounds[1], (self.n_fireworks, self.dim))
        self.firework_fitness = np.array([func(x) for x in self.fireworks])

    def explode_firework(self, firework, func):
        sparks = np.random.uniform(firework - self.alpha, firework + self.alpha, (self.n_sparks, self.dim))
        sparks_fitness = np.array([func(x) for x in sparks])
        return sparks, sparks_fitness

    def apply_mutation(self, sparks):
        mutated_sparks = sparks + np.random.normal(0, self.mutation_rate, sparks.shape)
        return np.clip(mutated_sparks, self.bounds[0], self.bounds[1])

    def update_fireworks(self, sparks, sparks_fitness):
        for i in range(self.n_fireworks):
            if i < len(sparks) and sparks_fitness[i] < self.firework_fitness[i]:
                self.fireworks[i] = sparks[i]
                self.firework_fitness[i] = sparks_fitness[i]

    def adapt_alpha(self, func):
        best_idx = np.argmin(self.firework_fitness)
        worst_idx = np.argmax(self.firework_fitness)
        self.alpha = self.alpha * (
            self.firework_fitness[best_idx] / (self.firework_fitness[worst_idx] + 1e-8)
        )

    def adapt_beta(self):
        self.beta = self.beta * 0.9

    def __call__(self, func):
        self.initialize_fireworks(func)

        for _ in range(int(self.budget / self.n_fireworks)):
            for i in range(self.n_fireworks):
                sparks, sparks_fitness = self.explode_firework(self.fireworks[i], func)
                mutated_sparks = self.apply_mutation(sparks)
                self.update_fireworks(mutated_sparks, sparks_fitness)

            self.adapt_alpha(func)
            self.adapt_beta()

            best_idx = np.argmin(self.firework_fitness)
            if self.firework_fitness[best_idx] < self.f_opt:
                self.f_opt = self.firework_fitness[best_idx]
                self.x_opt = self.fireworks[best_idx]

        return self.f_opt, self.x_opt
