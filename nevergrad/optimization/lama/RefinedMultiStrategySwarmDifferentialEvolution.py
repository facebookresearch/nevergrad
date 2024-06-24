import numpy as np


class RefinedMultiStrategySwarmDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality
        self.pop_size = 250  # Increased population size for enhanced exploration
        self.F_base = 0.6  # Base mutation factor
        self.CR = 0.8  # Crossover probability
        self.adapt_rate = 0.2  # Rate at which F adapts dynamically
        self.lambd = 0.85  # Control parameter for mutation strategy switching

    def __call__(self, func):
        # Initialize population within search space bounds
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Main evolutionary loop
        for i in range(int(self.budget / self.pop_size)):
            # Adaptive mutation factor influenced by a cosine curve for non-linear adaptation
            F_adapted = self.F_base + self.adapt_rate * np.cos(2 * np.pi * i / (self.budget / self.pop_size))

            for j in range(self.pop_size):
                # Dual mutation strategy controlled by dynamic parameter lambda
                if np.random.rand() < self.lambd:
                    # Strategy 1: DE/rand/2/bin
                    idxs = [idx for idx in range(self.pop_size) if idx != j]
                    a, b, c, d = pop[np.random.choice(idxs, 4, replace=False)]
                    mutant = a + F_adapted * (b - c) + F_adapted * (c - d)
                else:
                    # Strategy 2: DE/best/2/bin
                    idxs = [idx for idx in range(self.pop_size) if idx != j]
                    a, b = pop[np.random.choice(idxs, 2, replace=False)]
                    mutant = best_ind + F_adapted * (a - b) + F_adapted * (b - pop[j])

                mutant = np.clip(mutant, -5.0, 5.0)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[j])

                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[j]:
                    pop[j] = trial
                    fitness[j] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()

        return best_fitness, best_ind
