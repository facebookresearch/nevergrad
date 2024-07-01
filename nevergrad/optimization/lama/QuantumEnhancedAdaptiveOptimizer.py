import numpy as np


class QuantumEnhancedAdaptiveOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality of the problem
        self.pop_size = 100  # Increased population size for enhanced exploration
        self.F_min = 0.1  # Minimum differential weight
        self.F_max = 0.9  # Maximum differential weight
        self.CR = 0.8  # Fixed crossover probability
        self.q_influence = 0.15  # Quantum influence factor

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Quantum mutation and recombination loop
        for iteration in range(int(self.budget / self.pop_size)):
            # Adaptive differential weight based on iteration
            F = self.F_min + (self.F_max - self.F_min) * np.sin(
                np.pi * iteration / (self.budget / self.pop_size)
            )

            # Generate new trial vectors
            for i in range(self.pop_size):
                # Selection of mutation vector indices
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = pop[i] + F * (best_ind - pop[i]) + F * (a - b + c - pop[i])
                mutant = np.clip(mutant, -5.0, 5.0)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Quantum perturbation
                if np.random.rand() < self.q_influence:
                    trial += np.random.normal(0, 0.1, self.dim)

                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()

        return best_fitness, best_ind
