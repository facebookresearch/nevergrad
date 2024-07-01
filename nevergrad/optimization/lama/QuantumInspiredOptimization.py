import numpy as np


class QuantumInspiredOptimization:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0
        self.ub = 5.0
        self.alpha = 0.1  # Step size for update
        self.gamma = 0.05  # Step size for perturbation

    def __call__(self, func):
        self.f_opt = np.inf
        self.x_opt = None

        # Initialize a population of solutions
        population_size = 20
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        for t in range(1, self.budget // population_size + 1):
            # Select the best solution
            idx_best = np.argmin(fitness)
            x_best = population[idx_best].copy()

            # Update personal bests
            for i in range(population_size):
                if fitness[i] < self.f_opt:
                    self.f_opt = fitness[i]
                    self.x_opt = population[i].copy()

            # Quantum-inspired perturbation step
            for i in range(population_size):
                if np.random.rand() < 0.5:
                    direction = np.sign(np.random.randn(self.dim))
                    perturbation = self.gamma * direction * np.abs(x_best - population[i])
                    population[i] += perturbation
                else:
                    perturbation = self.alpha * (x_best - population[i])
                    population[i] += perturbation

            # Ensure solutions remain within bounds
            population = np.clip(population, self.lb, self.ub)

            # Evaluate fitness
            fitness = np.array([func(ind) for ind in population])

        return self.f_opt, self.x_opt
