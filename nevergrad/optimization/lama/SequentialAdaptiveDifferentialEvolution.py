import numpy as np


class SequentialAdaptiveDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.pop_size = 150  # Population size
        self.F_base = 0.5  # Base differential weight
        self.CR = 0.8  # Crossover probability

    def __call__(self, func):
        # Initialize population and fitness
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        # Track the best solution
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Evolution loop
        n_iterations = int(self.budget / self.pop_size)
        for iteration in range(n_iterations):
            # Adaptive F: decreases from 0.9 to 0.5 based on iteration count
            F_dynamic = self.F_base + (0.9 - self.F_base) * (1 - iteration / n_iterations)

            for i in range(self.pop_size):
                # Mutation and Crossover
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = a + F_dynamic * (b - c)
                mutant = np.clip(mutant, -5.0, 5.0)  # Ensure mutant is within bounds
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, pop[i])

                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()

        return best_fitness, best_ind
