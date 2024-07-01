import numpy as np


class ProgressiveHybridAdaptiveDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.pop_size = 200  # Increasing population size for more diverse initial sampling
        self.base_F = 0.5  # Base differential weight
        self.CR = 0.9  # Crossover probability
        self.F_increment = 0.1  # Increment factor for differential weight

    def __call__(self, func):
        # Initialize population and fitness array
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        # Track the best solution
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Evolution loop optimized within the given budget
        n_iterations = int(self.budget / self.pop_size)
        for iteration in range(n_iterations):
            # Progressive increase of F based on the half-life concept
            F_dynamic = self.base_F + self.F_increment * (1 - np.exp(-2 * iteration / n_iterations))
            for i in range(self.pop_size):
                # Mutation with DE/rand/1/bin strategy and progressive F
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = a + F_dynamic * (b - c)
                mutant = np.clip(mutant, -5.0, 5.0)  # Keep within bounds

                # Binomial Crossover with guaranteed change
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()

        return best_fitness, best_ind
