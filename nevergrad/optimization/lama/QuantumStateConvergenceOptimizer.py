import numpy as np


class QuantumStateConvergenceOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 100  # Population size for better manageability
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.quantum_influence = 0.1  # Probability of quantum mutation

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
                # Quantum Mutation influenced by best solution
                if np.random.rand() < self.quantum_influence:
                    mutation = best_ind + np.random.normal(0, 1, self.dim) * 0.1
                else:
                    # DE/rand/1 mutation strategy
                    indices = [idx for idx in range(self.pop_size) if idx != i]
                    a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                    mutation = pop[i] + self.F * (a - b + c - pop[i])

                mutation = np.clip(mutation, -5.0, 5.0)

                # Binomial crossover
                trial = np.where(np.random.rand(self.dim) < self.CR, mutation, pop[i])

                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[i]:
                    pop[i] = trial
                    fitness[i] = trial_fitness

                    # Update best solution
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()

        return best_fitness, best_ind
