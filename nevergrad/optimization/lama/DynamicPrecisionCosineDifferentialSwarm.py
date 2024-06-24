import numpy as np


class DynamicPrecisionCosineDifferentialSwarm:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 100  # Population size further reduced for higher precision
        self.F_base = 0.5  # Lower base mutation factor for very fine adjustments
        self.CR_base = 0.8  # Slightly lower base crossover probability
        self.adaptive_F_amplitude = 0.25  # Amplitude for the mutation factor oscillation
        self.adaptive_CR_amplitude = 0.15  # Amplitude for the crossover rate oscillation
        self.epsilon = 1e-10  # To avoid division by zero

    def __call__(self, func):
        # Initialize population within the bounds
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Main loop over the budget
        for i in range(int(self.budget / self.pop_size)):
            # Dynamic mutation and crossover factors using cosine modulation
            iteration_ratio = i / (self.budget / self.pop_size + self.epsilon)
            F = self.F_base + self.adaptive_F_amplitude * np.cos(2 * np.pi * iteration_ratio)
            CR = self.CR_base + self.adaptive_CR_amplitude * np.sin(2 * np.pi * iteration_ratio)

            for j in range(self.pop_size):
                # Mutation strategy: DE/rand/1/bin with dynamic F
                idxs = [idx for idx in range(self.pop_size) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = a + F * (b - c)

                # Clip to ensure staying within bounds
                mutant = np.clip(mutant, -5.0, 5.0)

                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[j])

                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[j]:
                    pop[j] = trial
                    fitness[j] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()

        return best_fitness, best_ind
