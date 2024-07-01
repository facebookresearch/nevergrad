import numpy as np


class QuantumAcceleratedEvolutionStrategy:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality of the problem
        self.pop_size = 150  # Increased population size for better exploration
        self.sigma_initial = 1.0  # Initial standard deviation for the mutation
        self.F_min = 0.1  # Minimum differential weight
        self.F_max = 0.9  # Maximum differential weight
        self.CR = 0.7  # Crossover probability
        self.q_impact = 0.1  # Quantum impact in mutation
        self.sigma_decay = 0.999  # Decay rate for sigma

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        sigma = self.sigma_initial

        # Evolution loop
        for iteration in range(int(self.budget / self.pop_size)):
            # Adapt sigma
            sigma *= self.sigma_decay

            # Adaptive differential weight based on iteration
            F = self.F_min + (self.F_max - self.F_min) * np.cos(
                np.pi * iteration / (self.budget / self.pop_size)
            )

            # Generate new trial vectors
            for i in range(self.pop_size):
                # Mutation using differential evolution strategy and quantum impact
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = best_ind + F * (a - b) + sigma * np.random.randn(self.dim)
                mutant += self.q_impact * np.random.standard_cauchy(self.dim)  # Quantum influenced mutation
                mutant = np.clip(mutant, -5.0, 5.0)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Evaluate
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()

        return best_fitness, best_ind
