import numpy as np


class AdaptiveOscillatoryCrossoverDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 100  # Adjusted population size for balance between exploration and exploitation
        self.F = 0.5  # Mutation factor initially set to 0.5
        self.CR_init = 0.5  # Initial crossover probability is moderate
        self.CR_final = 0.1  # Final crossover probability is low to focus search later
        self.alpha = 0.1  # Adaptive factor for mutation rate
        self.mutation_strategy = "best/2/bin"  # Mutation strategy using the best individual

    def __call__(self, func):
        # Initial population uniformly distributed within the search space
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        # Tracking the best solution found
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Evolution loop
        n_iterations = int(self.budget / self.pop_size)
        for iteration in range(n_iterations):
            # Oscillatory crossover rate and adaptive mutation factor
            CR = self.CR_final + (self.CR_init - self.CR_final) * np.cos(np.pi * iteration / n_iterations)
            F = self.F + self.alpha * np.sin(np.pi * iteration / n_iterations)

            for i in range(self.pop_size):
                # Mutation using best/2/bin strategy
                idxs = np.random.choice([idx for idx in range(self.pop_size) if idx != i], 3, replace=False)
                a, b, c = pop[idxs]
                mutant = best_ind + F * (a - b + c - best_ind)

                # Clipping to ensure individuals stay within bounds
                mutant = np.clip(mutant, -5.0, 5.0)

                # Crossover
                trial = np.where(np.random.rand(self.dim) < CR, mutant, pop[i])

                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()

        return best_fitness, best_ind
