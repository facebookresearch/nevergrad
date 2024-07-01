import numpy as np


class EnhancedDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.pop_size = 100  # Increased population size for better diversity
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        # Initialize population randomly
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        # Best solution found
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Evolutionary loop
        n_iterations = int(self.budget / self.pop_size)
        for iteration in range(n_iterations):
            for i in range(self.pop_size):
                # Adaptive mutation strategy: DE/current-to-best/1
                # Select indices for mutation (excluding current index i)
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b = pop[np.random.choice(idxs, 2, replace=False)]
                best = pop[best_idx]

                # Mutation: Including information from the current best
                mutant = np.clip(pop[i] + self.F * (best - pop[i]) + self.F * (a - b), -5.0, 5.0)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    pop[i] = trial
                    # Update best solution if found
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()
                        best_idx = i  # Update index of the best individual

            # Dynamic adaptation of parameters
            self.F = 0.5 + 0.3 * np.log1p(iteration) / np.log1p(n_iterations)
            self.CR = 0.5 + 0.4 * iteration / n_iterations

        return best_fitness, best_ind
