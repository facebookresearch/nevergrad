import numpy as np


class RefinedAdaptiveSwarmDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality
        self.pop_size = 300  # Population size maintained as 300
        self.F_base = 0.5  # Base mutation factor
        self.CR = 0.9  # Crossover probability
        self.adapt_rate = 0.1  # Adaptation rate for mutation factor

    def __call__(self, func):
        # Initialize population within the bounds
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Main loop over the budget
        for i in range(int(self.budget / self.pop_size)):
            F_adaptive = self.F_base + self.adapt_rate * np.sin(i / (self.budget / self.pop_size) * np.pi)

            for j in range(self.pop_size):
                # Mutation strategy: DE/rand/1/bin
                idxs = [idx for idx in range(self.pop_size) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = pop[j] + F_adaptive * (a - b) + F_adaptive * (c - pop[j])

                # Clip to ensure staying within bounds
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
