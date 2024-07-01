import numpy as np


class DualStrategyDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.pop_size = 200  # Population size optimized for balance
        self.F = 0.6  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.p = 0.1  # Probability of choosing best strategy

    def __call__(self, func):
        # Initialize population uniformly within the bounds
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(x) for x in pop])

        # Get the initial best solution
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Iteration loop
        n_iterations = int(self.budget / self.pop_size)
        for iteration in range(n_iterations):
            for i in range(self.pop_size):
                # Choose the mutation strategy
                if np.random.rand() < self.p:
                    # Best/1/bin strategy
                    indices = [idx for idx in range(self.pop_size) if idx != i]
                    a, b = pop[np.random.choice(indices, 2, replace=False)]
                    mutant = best_ind + self.F * (a - b)
                else:
                    # Rand/1/bin strategy
                    indices = [idx for idx in range(self.pop_size) if idx != i]
                    a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                    mutant = a + self.F * (b - c)

                # Clip within bounds
                mutant = np.clip(mutant, -5.0, 5.0)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Evaluate
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    pop[i] = trial
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()

        return best_fitness, best_ind
