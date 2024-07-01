import numpy as np


class GradientGuidedDifferentialEvolution:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.pop_size = 300  # Adjusted population size for better coverage
        self.F_base = 0.5  # Base mutation factor
        self.F_max = 0.8  # Maximum mutation factor, reduced to control variation
        self.CR = 0.85  # Crossover probability
        self.alpha = 0.1  # Gradient exploitation factor

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(x) for x in pop])

        # Find the initial best solution
        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Evolution loop within the budget constraint
        n_iterations = int(self.budget / self.pop_size)
        for iteration in range(n_iterations):
            # Calculate dynamic mutation factor with linear decay over iterations
            F_dynamic = self.F_base + (self.F_max - self.F_base) * (1 - iteration / n_iterations)
            for i in range(self.pop_size):
                # Mutation using DE/rand/1/bin strategy
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = a + F_dynamic * (b - c)
                mutant = np.clip(mutant, -5.0, 5.0)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                # Gradient exploitation
                grad_direction = best_ind - pop[i]
                grad_step = self.alpha * grad_direction
                trial += grad_step
                trial = np.clip(trial, -5.0, 5.0)

                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    pop[i] = trial
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()

        return best_fitness, best_ind
