import numpy as np


class AdaptivePerturbationDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.pop_size = 150  # Population size adjusted for efficiency
        self.F_base = 0.5  # Base differential weight
        self.CR = 0.7  # Crossover probability
        self.adapt_rate = 0.1  # Rate at which F is adapted

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
            F = self.F_base + self.adapt_rate * np.sin(iteration / n_iterations * np.pi)
            for i in range(self.pop_size):
                # Rand/1/bin strategy with adaptive perturbation
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = a + F * (b - c)

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
