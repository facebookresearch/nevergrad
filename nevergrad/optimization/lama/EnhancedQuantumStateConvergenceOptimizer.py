import numpy as np


class EnhancedQuantumStateConvergenceOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 50  # Reduced population size to increase refinement per individual
        self.F = 0.7  # Slightly reduced differential weight to stabilize convergence
        self.CR = 0.85  # Slightly reduced crossover probability to maintain good traits
        self.q_influence = 0.15  # Higher quantum influence to enhance exploration

    def __call__(self, func):
        # Initialize population
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Main optimization loop
        for _ in range(int(self.budget / self.pop_size)):
            for i in range(self.pop_size):
                # Apply quantum mutation with higher frequency and influence
                if np.random.rand() < self.q_influence:
                    mutation = best_ind + np.random.normal(0, 1, self.dim) * 0.15
                else:
                    # DE/rand/1 mutation strategy with modified random selection logic
                    indices = [idx for idx in range(self.pop_size) if idx != i]
                    a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                    mutation = a + self.F * (b - c)  # Focused on the difference for mutation

                mutation = np.clip(mutation, -5.0, 5.0)

                # Binomial crossover
                trial = np.where(np.random.rand(self.dim) < self.CR, mutation, pop[i])

                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                    # Update best if necessary
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()

        return best_fitness, best_ind
