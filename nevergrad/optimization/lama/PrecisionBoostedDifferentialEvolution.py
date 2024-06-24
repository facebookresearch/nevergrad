import numpy as np


class PrecisionBoostedDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 100  # Increased population for enhanced diversity
        self.F_base = 0.9  # Increased base mutation factor for aggressive exploration
        self.CR_base = 0.8  # Base crossover probability
        self.F_min = 0.2  # Higher minimum mutation factor to avoid too low exploration at later stages
        self.CR_min = 0.5  # Minimum crossover rate to maintain a decent level of recombination throughout
        self.F_scaling = 0.97  # Decay scaling for F
        self.CR_scaling = 0.985  # Decay scaling for CR

    def __call__(self, func):
        # Initialize population uniformly within search space bounds
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        # Identify the best individual
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx]

        # Evolutionary process given the budget constraints
        n_iterations = int(self.budget / self.pop_size)
        F = self.F_base
        CR = self.CR_base

        for iteration in range(n_iterations):
            # Decay mutation and crossover probabilities
            F = max(self.F_min, F * self.F_scaling)
            CR = max(self.CR_min, CR * self.CR_scaling)

            for i in range(self.pop_size):
                # Mutation strategy: 'rand/1/bin'
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = a + F * (b - c)

                # Clipping to bounds
                mutant = np.clip(mutant, -5.0, 5.0)

                # Crossover
                trial = np.array([mutant[j] if np.random.rand() < CR else pop[i][j] for j in range(self.dim)])

                # Evaluate trial solution
                trial_fitness = func(trial)

                # Selection
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                    # Update best solution found
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()

        return best_fitness, best_ind
