import numpy as np


class QuantumAdaptiveConvergenceOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 80  # Adjusted population size for more diversity
        self.F_base = 0.5  # Base differential weight
        self.CR_base = 0.9  # Base crossover probability
        self.q_influence_base = 0.05  # Base quantum influence
        self.q_influence_max = 0.25  # Max quantum influence
        self.adaptation_rate = 0.01  # Rate of parameter adaptation

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        F = self.F_base
        CR = self.CR_base
        q_influence = self.q_influence_base

        # Main optimization loop
        for iteration in range(int(self.budget / self.pop_size)):
            for i in range(self.pop_size):
                # Adaptively adjust F and CR based on iteration progress
                F = self.F_base + (0.8 - self.F_base) * (iteration / (self.budget / self.pop_size))
                CR = self.CR_base - (self.CR_base - 0.5) * (iteration / (self.budget / self.pop_size))
                q_influence = self.q_influence_base + (self.q_influence_max - self.q_influence_base) * np.sin(
                    np.pi * iteration / (self.budget / self.pop_size)
                )

                # Quantum-driven mutation
                if np.random.rand() < q_influence:
                    mutation = best_ind + np.random.normal(0, 1, self.dim) * (
                        0.1 + 0.2 * iteration / (self.budget / self.pop_size)
                    )
                else:
                    indices = [idx for idx in range(self.pop_size) if idx != i]
                    a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                    mutation = a + F * (b - c)

                mutation = np.clip(mutation, -5.0, 5.0)

                # Binomial crossover
                trial = np.where(np.random.rand(self.dim) < CR, mutation, pop[i])

                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()

        return best_fitness, best_ind
