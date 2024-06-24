import numpy as np


class QuantumAssistedHybridOptimizerV1:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = np.full(self.dim, -5.0)
        self.ub = np.full(self.dim, 5.0)

    def __call__(self, func):
        population_size = 150
        elite_size = 15
        evaluations = 0
        mutation_factor = 0.75  # Adjusted mutation factor for exploration
        crossover_probability = 0.8  # Medium level crossover probability
        quantum_probability = 0.02  # Initial quantum probability
        convergence_threshold = 1e-6  # Fine-grained threshold for detecting stagnation

        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations += population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        previous_best = np.inf

        while evaluations < self.budget:
            if abs(previous_best - self.f_opt) < convergence_threshold:
                mutation_factor *= 0.85  # Decrease mutation factor to intensify search
            previous_best = self.f_opt

            # Quantum-inspired exploration
            if np.random.rand() < quantum_probability:
                quantum_individual = np.random.uniform(self.lb, self.ub, self.dim)
                quantum_fitness = func(quantum_individual)
                evaluations += 1

                if quantum_fitness < self.f_opt:
                    self.f_opt = quantum_fitness
                    self.x_opt = quantum_individual

            # Differential evolution step
            new_population = []
            for i in range(population_size):
                idxs = [idx for idx in range(population_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = population[a] + mutation_factor * (population[b] - population[c])
                mutant = np.clip(mutant, self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < crossover_probability
                trial = np.where(cross_points, mutant, population[i])

                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    new_population.append(trial)
                    fitness[i] = trial_fitness

                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial

                else:
                    new_population.append(population[i])

            population = np.array(new_population)

            # Dynamically adjust the quantum probability
            if evaluations % 1000 == 0:
                quantum_probability = min(
                    0.1, quantum_probability + 0.01
                )  # Gradually increase the quantum probability

        return self.f_opt, self.x_opt
