import numpy as np


class RefinedTemporalAdaptiveDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 80  # Adjusted population size for better population diversity
        self.F_base = 0.8  # Initial mutation factor
        self.CR = 0.9  # Crossover probability
        self.F_min = 0.1  # Minimum mutation factor
        self.F_decay = 0.985  # Adjusted decay factor for mutation rate

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
            F = max(self.F_min, F * self.F_decay)  # Ensure F does not go below F_min

            for i in range(self.pop_size):
                # Differential evolution strategy: 'best/1/bin'
                idxs = np.random.choice([idx for idx in range(self.pop_size) if idx != i], 2, replace=False)
                b, c = pop[idxs]
                mutant = pop[best_idx] + F * (b - c)

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
