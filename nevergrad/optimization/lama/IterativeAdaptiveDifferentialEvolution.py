import numpy as np


class IterativeAdaptiveDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.pop_size = 200  # Increased population for wider exploration
        self.F_min = 0.1  # Minimum differential weight
        self.F_max = 0.9  # Maximum differential weight
        self.CR = 0.9  # High crossover probability for more trial vectors

    def __call__(self, func):
        # Initialize population uniformly between bounds
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        # Identify the best individual
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Evolutionary loop over the budget
        n_iterations = int(self.budget / self.pop_size)
        for iteration in range(n_iterations):
            F_dynamic = self.F_min + (self.F_max - self.F_min) * (n_iterations - iteration) / n_iterations

            for i in range(self.pop_size):
                # Select distinct indices excluding the target index i
                candidates = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(candidates, 3, replace=False)]

                # Mutant vector generation with adaptive F
                mutant = np.clip(a + F_dynamic * (b - c), -5.0, 5.0)

                # Crossover to generate trial vector
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, pop[i])

                # Evaluate trial vector
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()

        return best_fitness, best_ind
