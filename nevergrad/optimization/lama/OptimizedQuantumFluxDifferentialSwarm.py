import numpy as np


class OptimizedQuantumFluxDifferentialSwarm:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 1000  # Population size tailored to the search space
        self.F = 0.7  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.quantum_probability = 0.2  # Probability of quantum mutation
        self.learning_rate = 0.1  # Learning rate for quantum mutation refinement
        self.adaptation_factor = 0.05  # Smoothing factor for adapting search dynamics

    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        for i in range(int(self.budget / self.pop_size)):
            # Adaptation of parameters based on search progress
            phase = i / (self.budget / self.pop_size)
            F = self.F + self.adaptation_factor * np.sin(np.pi * phase)
            CR = self.CR + self.adaptation_factor * np.cos(np.pi * phase)

            for j in range(self.pop_size):
                if np.random.rand() < self.quantum_probability:
                    # Quantum mutation
                    mean_state = best_ind + self.learning_rate * (pop[j] - best_ind)
                    scale = self.learning_rate * np.abs(pop[j] - best_ind)
                    mutation = np.random.normal(mean_state, scale)
                    mutation = np.clip(mutation, -5.0, 5.0)
                else:
                    # DE/rand/1/bin strategy
                    indices = [idx for idx in range(self.pop_size) if idx != j]
                    a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                    mutation = a + F * (b - c)
                    mutation = np.clip(mutation, -5.0, 5.0)

                # Crossover
                trial = np.where(np.random.rand(self.dim) < CR, mutation, pop[j])

                # Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[j]:
                    pop[j] = trial
                    fitness[j] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()

        return best_fitness, best_ind
