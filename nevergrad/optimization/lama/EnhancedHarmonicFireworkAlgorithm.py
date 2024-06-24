import numpy as np


class EnhancedHarmonicFireworkAlgorithm:
    def __init__(self, budget=10000, n_fireworks=20, n_sparks=10, alpha=0.5, beta=2.0, mutation_rate=0.1):
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

    def adapt_parameters(self):
        self.alpha = self.alpha * 0.95
        self.beta = self.beta * 0.9
        self.mutation_rate = max(self.mutation_rate * 0.9, 0.01)

    def local_search(self, func):
        for i in range(self.n_fireworks):
            best_firework = self.fireworks[i].copy()
            best_fitness = self.firework_fitness[i]

            for _ in range(3):
                new_firework = self.fireworks[i] + np.random.normal(0, 0.1, self.dim)
                new_fitness = func(new_firework)

                if new_fitness < best_fitness:
                    best_firework = new_firework
                    best_fitness = new_fitness

            self.fireworks[i] = best_firework
            self.firework_fitness[i] = best_fitness

    def __call__(self, func):
        self.initialize_fireworks(func)

        for _ in range(int(self.budget / self.n_fireworks)):
            for i in range(self.n_fireworks):
                sparks, sparks_fitness = self.explode_firework(self.fireworks[i], func)
                mutated_sparks = self.apply_mutation(sparks)
                self.update_fireworks(mutated_sparks, sparks_fitness)

            self.adapt_parameters()
            self.local_search(func)

            best_idx = np.argmin(self.firework_fitness)
            if self.firework_fitness[best_idx] < self.f_opt:
                self.f_opt = self.firework_fitness[best_idx]
                self.x_opt = self.fireworks[best_idx]

        return self.f_opt, self.x_opt
