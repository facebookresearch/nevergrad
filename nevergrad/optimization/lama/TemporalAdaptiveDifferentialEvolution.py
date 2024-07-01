import numpy as np


class TemporalAdaptiveDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 50  # Population size reduced for more focused search
        self.F_base = 0.5  # Base mutation factor
        self.CR = 0.9  # Crossover probability
        self.F_decay = 0.99  # Decay factor for mutation rate

    def __call__(self, func):
        # Initialize population uniformly within bounds
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        # Track the best solution found
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Evolution loop
        n_iterations = int(self.budget / self.pop_size)
        F = self.F_base
        for iteration in range(n_iterations):
            # Temporally decaying mutation factor
            F *= self.F_decay

            for i in range(self.pop_size):
                # Mutation strategy: 'rand/1/bin'
                idxs = np.random.choice([idx for idx in range(self.pop_size) if idx != i], 3, replace=False)
                a, b, c = pop[idxs]
                mutant = pop[i] + F * (a - b + c - pop[i])

                # Clipping to bounds
                mutant = np.clip(mutant, -5.0, 5.0)

                # Crossover
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
