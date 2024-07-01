import numpy as np


class EnhancedQuantumSymbioticStrategyV5:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Given problem dimensionality
        self.lb = -5.0 * np.ones(self.dim)  # Lower bound of the search space
        self.ub = 5.0 * np.ones(self.dim)  # Upper bound of the search space

    def __call__(self, func):
        population_size = 500
        elite_size = 50
        evaluations = 0
        mutation_factor = 0.9
        crossover_probability = 0.7
        quantum_probability = 0.1
        adaptive_scaling_factor = lambda t: 0.2 * np.exp(-0.2 * t)

        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations += population_size

        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        while evaluations < self.budget:
            current_best_fitness = np.min(fitness)

            # Quantum mutation step with fine-tuned control
            if np.random.rand() < quantum_probability:
                elite_indices = np.argsort(fitness)[:elite_size]
                for i in elite_indices:
                    if evaluations >= self.budget:
                        break
                    time_factor = evaluations / self.budget
                    quantum_mutant = population[i] + np.random.normal(
                        0, adaptive_scaling_factor(time_factor), self.dim
                    )
                    quantum_mutant = np.clip(quantum_mutant, self.lb, self.ub)
                    quantum_fitness = func(quantum_mutant)
                    evaluations += 1

                    if quantum_fitness < fitness[i]:
                        population[i] = quantum_mutant
                        fitness[i] = quantum_fitness
                        if quantum_fitness < self.f_opt:
                            self.f_opt = quantum_fitness
                            self.x_opt = quantum_mutant

            # Enhanced symbiotic mutation and crossover with dynamic elite selection
            for i in range(population_size):
                if evaluations >= self.budget:
                    break
                inds = np.random.choice(population_size, 3, replace=False)
                x1, x2, x3 = population[inds]

                # Mutation and Crossover
                mutant = x1 + mutation_factor * (x2 - x3)
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
