import numpy as np


class EnhancedAdaptiveSinusoidalDifferentialSwarm:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 200  # Increased population size for enhanced exploration
        self.F_base = 0.5  # Base mutation factor
        self.CR_base = 0.95  # Increased base crossover probability for better trial acceptance
        self.adaptive_F_amplitude = 0.45  # Increased mutation factor amplitude for enhanced adaptive range
        self.adaptive_CR_amplitude = (
            0.25  # Increased crossover rate amplitude for robust exploration/exploitation balance
        )

    def __call__(self, func):
        # Initialize population within the bounds
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Main loop over the budget
        for i in range(int(self.budget / self.pop_size)):
            # Dynamic mutation and crossover factors using enhanced sinusoidal modulation
            iteration_ratio = i / (self.budget / self.pop_size)
            F = self.F_base + self.adaptive_F_amplitude * np.sin(np.pi * iteration_ratio)
            CR = self.CR_base + self.adaptive_CR_amplitude * np.cos(np.pi * iteration_ratio)

            for j in range(self.pop_size):
                # Mutation: DE/rand/1/bin with adaptive F
                idxs = [idx for idx in range(self.pop_size) if idx != j]
                a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                mutant = a + F * (b - c)
                mutant = np.clip(mutant, -5.0, 5.0)  # Ensure boundaries are respected

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
