import numpy as np


class AdaptiveSineCosineDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.pop_size = 250  # Adjusted population size for better diversity
        self.F_base = 0.5  # Base factor for mutation
        self.F_max = 0.9  # Maximum factor for mutation
        self.CR = 0.8  # Crossover probability

    def __call__(self, func):
        # Initialize population and fitness
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(x) for x in pop])

        # Identify the best initial agent
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Evolution loop within the budget constraint
        n_iterations = int(self.budget / self.pop_size)
        for iteration in range(n_iterations):
            # Dynamically adjust F with a sine modulation to balance exploration and exploitation
            F_dynamic = self.F_base + (self.F_max - self.F_base) * np.sin(np.pi * iteration / n_iterations)
            for i in range(self.pop_size):
                # Mutation using DE/rand/1 strategy with dynamic F
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = a + F_dynamic * (b - c)
                mutant = np.clip(mutant, -5.0, 5.0)

                # Crossover: DE/binomial strategy
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    pop[i] = trial
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()

        return best_fitness, best_ind
