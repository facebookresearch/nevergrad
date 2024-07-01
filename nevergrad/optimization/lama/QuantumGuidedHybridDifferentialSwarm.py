import numpy as np


class QuantumGuidedHybridDifferentialSwarm:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 750  # Further increased population size for enhanced exploration and exploitation
        self.F_base = 0.6  # Slightly higher mutation rate for more aggressive search
        self.CR_base = 0.85  # Slightly lower crossover rate to maintain diversity in the population
        self.quantum_probability = 0.15  # Increased probability for quantum-driven mutation
        self.vortex_effect = 0.2  # Introducing a new vortex effect factor for complex landscape navigation

    def __call__(self, func):
        # Initialize population within bounds
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        # Main loop over the budget
        for i in range(int(self.budget / self.pop_size)):
            # Dynamic factors adjusted with a sine-cosine modulation for adaptive behavior
            iteration_ratio = i / (self.budget / self.pop_size)
            F = self.F_base + np.sin(2 * np.pi * iteration_ratio) * self.vortex_effect
            CR = self.CR_base + np.cos(2 * np.pi * iteration_ratio) * self.vortex_effect

            for j in range(self.pop_size):
                if np.random.rand() < self.quantum_probability:
                    # Quantum-inspired mutation using a complex Gaussian distribution
                    mean_quantum_state = best_ind + (pop[j] - best_ind) / 2
                    quantum_mutation = np.random.normal(
                        loc=mean_quantum_state, scale=np.abs(best_ind - pop[j])
                    )
                    quantum_mutation = np.clip(quantum_mutation, -5.0, 5.0)
                    mutant = quantum_mutation
                else:
                    # Classical DE mutation: DE/rand-to-best/1
                    idxs = [idx for idx in range(self.pop_size) if idx != j]
                    a, b = pop[np.random.choice(idxs, 2, replace=False)]
                    mutant = pop[j] + F * (best_ind - pop[j]) + F * (a - b)
                    mutant = np.clip(mutant, -5.0, 5.0)  # Enforce boundary constraints

                # Crossover operation
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[j])

                # Selection process
                trial_fitness = func(trial)
                if trial_fitness < fitness[j]:
                    pop[j] = trial
                    fitness[j] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()

        return best_fitness, best_ind
