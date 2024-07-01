import numpy as np


class AdaptiveResilientQuantumCrossoverStrategy:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed to 5 as per the problem description
        self.lb = -5.0 * np.ones(self.dim)  # Lower bounds
        self.ub = 5.0 * np.ones(self.dim)  # Upper bounds

    def __call__(self, func):
        population_size = 100  # Manageable population size
        elite_size = 10  # Number of top performers
        evaluations = 0
        mutation_factor = 0.5  # Starting mutation factor
        crossover_probability = 0.7  # Probability of crossover
        quantum_probability = 0.1  # Chance of quantum-informed mutation

        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations += population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        # Evolution loop
        while evaluations < self.budget:
            # Quantum step
            quantum_mutants = population[:elite_size] + np.random.normal(0, 0.1, (elite_size, self.dim))
            quantum_mutants = np.clip(quantum_mutants, self.lb, self.ub)
            quantum_fitness = np.array([func(ind) for ind in quantum_mutants])
            evaluations += elite_size

            # Implement elitism
            for i in range(elite_size):
                if quantum_fitness[i] < fitness[i]:
                    population[i] = quantum_mutants[i]
                    fitness[i] = quantum_fitness[i]

            # Genetic operators
            for i in range(population_size):
                if evaluations >= self.budget:
                    break
                # Tournament selection
                indices = np.random.choice(population_size, 3, replace=False)
                a, b, c = indices
                if fitness[a] < fitness[b]:
                    if fitness[a] < fitness[c]:
                        best = a
                    else:
                        best = c
                else:
                    if fitness[b] < fitness[c]:
                        best = b
                    else:
                        best = c

                mutant = population[best] + mutation_factor * (
                    population[a] - population[b] + population[c] - population[best]
                )
                mutant = np.clip(mutant, self.lb, self.ub)
                trial = np.where(np.random.rand(self.dim) < crossover_probability, mutant, population[i])
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial

        return self.f_opt, self.x_opt
