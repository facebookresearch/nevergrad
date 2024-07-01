import numpy as np


class StrategicDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.pop_size = 150  # Population size has been adjusted for enhanced exploration
        self.F_base = 0.5  # Base differential weight
        self.CR_base = 0.5  # Base crossover probability

    def __call__(self, func):
        # Initialize population randomly within bounds
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        # Track the best solution
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Main evolutionary loop
        n_iterations = int(self.budget / self.pop_size)
        for iteration in range(n_iterations):
            for i in range(self.pop_size):
                # Indices for mutation strategy (excluding current index i)
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]

                # Mutation strategy: DE/rand-to-best/2
                best = pop[best_idx]
                mutant = np.clip(
                    pop[i] + self.F_base * (best - pop[i]) + self.F_base * (a - b + c - pop[i]), -5.0, 5.0
                )

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR_base
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
                        best_idx = i

            # Adaptive mutation and crossover rates
            self.F_base = 0.1 + 0.7 * (1 - iteration / n_iterations)
            self.CR_base = 0.1 + 0.8 * (iteration / n_iterations)

        return best_fitness, best_ind
