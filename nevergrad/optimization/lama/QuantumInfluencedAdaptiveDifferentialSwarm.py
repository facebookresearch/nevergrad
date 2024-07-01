import numpy as np


class QuantumInfluencedAdaptiveDifferentialSwarm:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 500  # Further increased population size for enhanced exploration
        self.F_base = 0.5  # Adjusted base mutation factor for stability
        self.CR_base = 0.9  # Increased base crossover probability for stronger exploration
        self.adaptive_F_amplitude = 0.3  # Increased mutation amplitude for wider search capability
        self.adaptive_CR_amplitude = 0.3  # Increased CR amplitude for dynamic exploration
        self.quantum_probability = 0.1  # Probability of quantum-inspired mutation

    def __call__(self, func):
        # Initialize population within the bounds
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Main loop over the budget
        for i in range(int(self.budget / self.pop_size)):
            # Dynamic mutation and crossover factors with sinusoidal modulation
            iteration_ratio = i / (self.budget / self.pop_size)
            F = self.F_base + self.adaptive_F_amplitude * np.sin(2 * np.pi * iteration_ratio)
            CR = self.CR_base + self.adaptive_CR_amplitude * np.sin(2 * np.pi * iteration_ratio)

            for j in range(self.pop_size):
                if np.random.rand() < self.quantum_probability:
                    # Quantum-inspired mutation
                    quantum_mutation = np.random.normal(loc=best_ind, scale=np.abs(best_ind - pop[j]) / 2)
                    quantum_mutation = np.clip(quantum_mutation, -5.0, 5.0)
                    mutant = quantum_mutation
                else:
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
