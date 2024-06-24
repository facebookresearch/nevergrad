import numpy as np


class RefinedQuantumFluxDifferentialSwarm:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 1200  # Adjusted population size for more exploration
        self.F_base = 0.8  # Adjusted base factor for mutation for stronger mutations
        self.CR_base = 0.85  # Adjusted base crossover probability
        self.quantum_probability = 0.25  # Increased quantum-driven mutation probability
        self.vortex_factor = 0.25  # Modified vortex factor for dynamic strategy modulation
        self.epsilon = 1e-6  # Stability constant for quantum mutation

    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Main optimization loop
        for i in range(int(self.budget / self.pop_size)):
            # Update dynamic parameters
            iteration_ratio = i / (self.budget / self.pop_size)
            F = self.F_base + self.vortex_factor * np.sin(2 * np.pi * iteration_ratio)
            CR = self.CR_base - self.vortex_factor * np.cos(2 * np.pi * iteration_ratio)

            for j in range(self.pop_size):
                # Quantum-inspired mutation with variable probability
                if np.random.rand() < self.quantum_probability:
                    mean_quantum_state = best_ind + (pop[j] - best_ind) / 2
                    scale = (np.abs(best_ind - pop[j]) + self.epsilon) / 2
                    quantum_mutation = np.random.normal(mean_quantum_state, scale)
                    quantum_mutation = np.clip(quantum_mutation, -5.0, 5.0)
                    mutant = quantum_mutation
                else:
                    # DE mutation: DE/rand/1/bin with best influence
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
