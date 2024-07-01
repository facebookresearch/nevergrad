import numpy as np


class QuantumFluxDifferentialSwarm:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 1000  # Increased population size for greater exploration
        self.F_base = 0.5  # Base factor for mutation
        self.CR_base = 0.9  # High crossover probability to favor recombination
        self.quantum_probability = 0.2  # Higher probability for quantum-driven mutation
        self.vortex_factor = 0.3  # Enhanced vortex factor for dynamic strategy

    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Main optimization loop
        for i in range(int(self.budget / self.pop_size)):
            # Adjusting factors based on a dynamic non-linear modulation
            iteration_ratio = i / (self.budget / self.pop_size)
            F = self.F_base + self.vortex_factor * np.sin(np.pi * iteration_ratio)
            CR = self.CR_base - self.vortex_factor * np.cos(np.pi * iteration_ratio)

            for j in range(self.pop_size):
                # Quantum-inspired mutation with higher probability
                if np.random.rand() < self.quantum_probability:
                    mean_quantum_state = best_ind + (pop[j] - best_ind) / 2
                    scale = np.abs(best_ind - pop[j]) / 2
                    quantum_mutation = np.random.normal(mean_quantum_state, scale)
                    quantum_mutation = np.clip(quantum_mutation, -5.0, 5.0)
                    mutant = quantum_mutation
                else:
                    # Traditional DE mutation: DE/rand/1 with best influence
                    idxs = [idx for idx in range(self.pop_size) if idx != j]
                    a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
                    mutant = a + F * (b - c) + F * (best_ind - pop[j])
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
