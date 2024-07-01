import numpy as np


class HyperQuantumConvergenceOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality of the problem
        self.pop_size = 500  # Population size adjusted for focused search
        self.F = 0.5  # Differential weight, slightly reduced for stability
        self.CR = 0.8  # Crossover probability, adjusted to prevent premature convergence
        self.quantum_probability = 0.25  # Increased probability of quantum mutation
        self.learning_rate = 0.05  # Reduced learning rate for more subtle quantum adjustments
        self.adaptation_factor = 0.1  # Increased adaptation factor for dynamic response

    def __call__(self, func):
        pop = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        best_idx = np.argmin(fitness)
        best_fitness = fitness[best_idx]
        best_ind = pop[best_idx].copy()

        for i in range(int(self.budget / self.pop_size)):
            # Adjusting parameters based on phase of optimization
            phase = i / (self.budget / self.pop_size)
            F = self.F + self.adaptation_factor * np.sin(np.pi * phase * 2)  # Faster sinusoidal variation
            CR = self.CR + self.adaptation_factor * np.cos(np.pi * phase / 2)  # Slower cosine variation

            for j in range(self.pop_size):
                if np.random.rand() < self.quantum_probability:
                    # Enhanced quantum mutation
                    mean_state = best_ind + self.learning_rate * (pop[j] - best_ind)
                    scale = self.learning_rate * np.sqrt(np.abs(pop[j] - best_ind))
                    mutation = np.random.normal(mean_state, scale)
                    mutation = np.clip(mutation, -5.0, 5.0)
                else:
                    # Differential Evolution Mutation: DE/rand-to-best/1/bin
                    indices = [idx for idx in range(self.pop_size) if idx != j]
                    a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                    mutation = pop[j] + F * (best_ind - pop[j]) + F * (b - c)
                    mutation = np.clip(mutation, -5.0, 5.0)

                # Crossover
                trial = np.where(np.random.rand(self.dim) < CR, mutation, pop[j])

                # Fitness Evaluation and Selection
                trial_fitness = func(trial)
                if trial_fitness < fitness[j]:
                    pop[j] = trial
                    fitness[j] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()

        return best_fitness, best_ind
