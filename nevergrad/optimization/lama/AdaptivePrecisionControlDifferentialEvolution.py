import numpy as np


class AdaptivePrecisionControlDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 150  # Increased population size for more diversity
        self.F = 0.5  # Base mutation factor
        self.CR = 0.9  # Base crossover probability
        self.adaptive_F = True  # Flag to adaptively adjust F
        self.adaptive_CR = True  # Flag to adaptively adjust CR

    def __call__(self, func):
        # Initialize population within search space bounds
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        # Find the best initial solution
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Main evolutionary loop
        for i in range(int(self.budget / self.pop_size)):
            for j in range(self.pop_size):
                # Mutation: DE/rand/1/bin
                idxs = [idx for idx in range(self.pop_size) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = a + self.F * (b - c)
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

            # Adaptive strategy updates
            if self.adaptive_F:
                self.F = max(0.1, self.F * (0.99 if best_fitness < 1e-6 else 1.01))
            if self.adaptive_CR:
                self.CR = min(1.0, self.CR * (1.01 if best_fitness < 1e-4 else 0.99))

        return best_fitness, best_ind
