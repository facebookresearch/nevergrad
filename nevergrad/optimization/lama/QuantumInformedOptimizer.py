import numpy as np


class QuantumInformedOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5
        self.lower_bound = -5.0
        self.upper_bound = 5.0

    def __call__(self, func):
        # Optimization setup
        current_budget = 0
        population_size = 50  # Smaller population to manage computational resources
        mutation_factor = 0.8  # Initial mutation factor for exploration
        crossover_prob = 0.7  # Initial crossover probability for exploration

        # Initialize population
        population = np.random.uniform(self.lower_bound, self.upper_bound, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        current_budget += population_size

        best_index = np.argmin(fitness)
        best_solution = population[best_index]
        best_fitness = fitness[best_index]

        # Quantum-inspired phase to enhance exploration and exploitation
        quantum_beta = 1.0  # Quantum behavior parameter
        quantum_alpha = 0.01  # Quantum learning rate
        quantum_population = quantum_beta * np.random.randn(population_size, self.dim)
        quantum_population = np.clip(quantum_population + population, self.lower_bound, self.upper_bound)

        while current_budget < self.budget:
            new_population = np.empty_like(population)
            new_quantum_population = np.empty_like(quantum_population)
            for i in range(population_size):
                if current_budget >= self.budget:
                    break

                # Mutation and crossover phases for both classical and quantum populations
                indices = np.arange(population_size)
                indices = np.delete(indices, i)
                random_indices = np.random.choice(indices, 3, replace=False)
                x1, x2, x3 = population[random_indices]
                q1, q2, q3 = quantum_population[random_indices]

                mutant = population[i] + mutation_factor * (x1 - x2 + x3 - population[i])
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                quantum_mutant = quantum_population[i] + quantum_alpha * (
                    q1 - q2 + q3 - quantum_population[i]
                )
                quantum_mutant = np.clip(quantum_mutant, self.lower_bound, self.upper_bound)

                # Crossover
                trial = np.where(np.random.rand(self.dim) < crossover_prob, mutant, population[i])
                quantum_trial = np.where(
                    np.random.rand(self.dim) < crossover_prob, quantum_mutant, quantum_population[i]
                )

                trial_fitness = func(trial)
                quantum_trial_fitness = func(quantum_trial)
                current_budget += 2  # Incrementing for both classical and quantum evaluations

                # Selection
                if quantum_trial_fitness < trial_fitness:
                    trial_fitness = quantum_trial_fitness
                    trial = quantum_trial

                if trial_fitness < fitness[i]:
                    new_population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_solution = trial
                else:
                    new_population[i] = population[i]

                new_quantum_population[i] = quantum_trial

            population = new_population
            quantum_population = new_quantum_population

            # Adaptively adjust mutation and crossover parameters
            mutation_factor *= 0.995  # Gradual decrease
            crossover_prob *= 1.005  # Gradual increase
            quantum_alpha *= 0.99  # Reduce quantum impact over time

        return best_fitness, best_solution
