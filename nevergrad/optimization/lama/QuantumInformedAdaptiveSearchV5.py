import numpy as np


class QuantumInformedAdaptiveSearchV5:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5
        self.lb = -5.0 * np.ones(self.dim)
        self.ub = 5.0 * np.ones(self.dim)

    def __call__(self, func):
        population_size = 300
        elite_size = 30
        evaluations = 0
        mutation_factor = 0.9
        crossover_probability = 0.7
        quantum_probability = 0.15
        adaptive_rate = 0.03
        learning_period = 15

        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations += population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        previous_best = self.f_opt

        while evaluations < self.budget:
            # Quantum-Informed Mutation
            if np.random.rand() < quantum_probability:
                for i in range(elite_size):
                    if evaluations >= self.budget:
                        break
                    quantum_individual = population[i] + np.random.normal(
                        loc=0, scale=0.3 / np.sqrt(fitness[i] + 1), size=self.dim
                    )
                    quantum_individual = np.clip(quantum_individual, self.lb, self.ub)
                    quantum_fitness = func(quantum_individual)
                    evaluations += 1

                    if quantum_fitness < fitness[i]:
                        population[i] = quantum_individual
                        fitness[i] = quantum_fitness

                        if quantum_fitness < self.f_opt:
                            self.f_opt = quantum_fitness
                            self.x_opt = quantum_individual

            # Differential Evolution
            indices = np.random.permutation(population_size)
            for i in indices:
                if evaluations >= self.budget:
                    break
                idxs = [idx for idx in indices if idx != i][:3]
                a, b, c = population[idxs]
                mutant = np.clip(a + mutation_factor * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < crossover_probability
                trial = np.where(cross_points, mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness

                if trial_fitness < self.f_opt:
                    self.f_opt = trial_fitness
                    self.x_opt = trial

            # Adaptive adjustment of parameters
            if evaluations % learning_period == 0:
                if np.abs(previous_best - self.f_opt) < 1e-4:
                    mutation_factor *= 1 - adaptive_rate
                    crossover_probability *= 1 + adaptive_rate
                else:
                    mutation_factor = min(mutation_factor * (1 + adaptive_rate), 1.0)
                    crossover_probability = max(crossover_probability * (1 - adaptive_rate), 0.3)
                previous_best = self.f_opt

        return self.f_opt, self.x_opt
