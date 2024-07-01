import numpy as np


class OscillatoryCrossoverDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # The dimensionality of the problem
        self.pop_size = 120  # Adjust population size for a good balance between exploration and exploitation
        self.F = 0.8  # Mutation factor
        self.CR_init = 0.9  # Initial crossover probability
        self.CR_final = 0.1  # Final crossover probability
        self.mutation_strategy = "rand/2/bin"  # Using two difference vectors for mutation

    def __call__(self, func):
        # Initial population uniformly distributed within the search space
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        # Tracking the best solution
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Evolution loop
        n_iterations = int(self.budget / self.pop_size)
        for iteration in range(n_iterations):
            # Oscillatory crossover rate
            CR = self.CR_final + (self.CR_init - self.CR_final) * np.cos(np.pi * iteration / n_iterations)

            for i in range(self.pop_size):
                # Mutation using rand/2/bin strategy
                idxs = np.random.choice([idx for idx in range(self.pop_size) if idx != i], 4, replace=False)
                a, b, c, d = pop[idxs]
                mutant = a + self.F * (b - c + d - pop[i])

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
